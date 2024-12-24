import streamlit as st
import tempfile
import os
import glob
import base64
from utils import get_base64_image, load_css_with_image, load_text_from_file
from config import opt

AUDIO_STYLE_DIR = opt.AUDIO_STYLE_DIR
AVAILABLE_STORY_DIR = opt.AVAILABLE_STORY_DIR
OUTPUT_DIR = opt.OUTPUT_DIR
ACCEPT_EXTENSION = opt.accept_extetion

# Streamlit UI
st.set_page_config(page_title="Text-to-Speech Generator", layout="wide")
st.title("Text to speech")

# Load and encode the image
image_path = "static/clound.avif"  # Replace with your image file name
base64_image = get_base64_image(image_path)

if base64_image:
    page_bg_img = load_css_with_image("static/style.css", image_path)
    # Inject the custom CSS
    st.markdown(page_bg_img, unsafe_allow_html=True)

available_stories_path = glob.glob(f'{AVAILABLE_STORY_DIR}/*.txt')
available_stories = [i[:-4] for i in os.listdir(AVAILABLE_STORY_DIR)]

# Predefined text files
PREDEFINED_FILES = dict(zip(available_stories, available_stories_path))

# Section 1: Choose Input Method
input_method = st.radio("Select an input method:", ["User Input", "Available story"], horizontal=True)

# Input Text Section
if input_method == "User Input":
    input_text = st.text_area("Type your text here:", placeholder="Type something to convert into speech...", height=150)
else:
    # st.info("Choose a predefined text file to use as input.",)
    predefined_option = st.selectbox("Select a story:", list(PREDEFINED_FILES.keys()))
    input_text = load_text_from_file(PREDEFINED_FILES[predefined_option])
    st.text_area("Text Content:", input_text, height=150, disabled=True)


############################################################################################################
st.divider()
st.header("2. Select Language")
st.info("Choose a language for Text-to-Speech generation.")
available_languages = {
    "English": "en",
    # "French": "fr",
    # "Spanish": "es",
    # "German": "de",
    # "Chinese": "zh",
    # "Japanese": "ja",
    # "Korean": "ko"
}
selected_language = st.selectbox("Select a language:", list(available_languages.keys()))

# Map the selected language to its language code
lang_code = available_languages[selected_language]

############################################################################################################
st.divider()
# Display audio styles as a dropdown list
st.header("3. Choose Audio Style")

selected_style = None

audio_styles = [item for item in os.listdir(AUDIO_STYLE_DIR) if os.path.isdir(os.path.join(AUDIO_STYLE_DIR, item))]
col1, col2 = st.columns((3,7))
with col1:
    selected_style = st.selectbox("Select an audio style:", audio_styles)
    st.success(f"Selected style: {selected_style}")

sample_styles = os.listdir(AUDIO_STYLE_DIR + '/' + selected_style)
with col2:
    selected_style_file = st.selectbox("Select an sample style:", sample_styles)
    selected_audio = os.path.join(AUDIO_STYLE_DIR,selected_style, selected_style_file)
    st.audio(selected_audio, format="audio/wav")
    st.success(f"Selected style: {selected_style_file}")

st.divider()
############################################################################################################
import os
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf
import sys
from pydub import AudioSegment
import nltk
from nltk.tokenize import sent_tokenize

@st.cache_resource
def ntlk_download():
    nltk.download('punkt')
    nltk.download('punkt_tab')
ntlk_download()


## TTS Model Initialization (Load Once)
@st.cache_resource
def load_tts_model():
    repo_path = "XTTS-v2"
    config_path = os.path.join(repo_path, "config.json")
    speaker_file_path = os.path.join(repo_path, "speakers_xtts.pth")
    checkpoint_dir = repo_path

    # Load configuration
    config = XttsConfig()
    config.load_json(config_path)

    # Initialize and load model
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, eval=True)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, config, device

model, config, device = load_tts_model()

# TTS processing function
def process_tts(input_text, speaker_audio, lang, output_file):
    print(f"Processing TTS for input: {input_text}")
    outputs = model.synthesize(
        input_text,
        config,
        speaker_wav=speaker_audio,
        gpt_cond_len=3,
        language=lang,
    )
    audio_data = outputs["wav"]
    sample_rate = 24000
    sf.write(output_file, audio_data, sample_rate)
    print(f"Audio saved successfully at {output_file}")


def generate_long_audio(input_story, style_audio, save_file, lang='en'):

    sentences = sent_tokenize(input_story)
    audio_files = []
    for i, sentence in enumerate(sentences):
        output_file = f"audio_sentence_{i}.wav"
        process_tts(sentence, style_audio,lang, output_file)
        audio_files.append(output_file)

    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=500)  

    for audio_file in audio_files:
        audio = AudioSegment.from_wav(audio_file)
        combined += audio + silence

    combined.export(save_file, format="wav")

    for audio_file in audio_files:
        os.remove(audio_file)
############################################################################################################

# Section 3: Generate Audio
# st.divider()
st.header("4. Generate and Play Audio")
selected_style = AUDIO_STYLE_DIR + '/' +st.session_state.selected_style

# Create a full-width button using Streamlit columns
if st.button("Generate Audio"):

    if input_text and selected_style:

        ############################################################################################################
        # Generate WAV file
        speaker_audio = selected_style

        output_filename = "output.wav"
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        # Process TTS
        with st.spinner("Generating audio... Please wait."):
            generate_long_audio(input_text, speaker_audio, save_path, lang=lang_code)

        ############################################################################################################
        # Show success message and play audio
        st.success(f"Audio generated successfully with style: **{selected_style}**. Playing below:")
        st.audio(save_path, format="audio/wav")

        # os.unlink(wav_file)  # Clean up temporary file
    elif not input_text:
        st.error("Please enter or select text first.")
    elif not selected_style:
        st.error("Please choose an audio style.")
