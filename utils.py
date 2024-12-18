import base64
import soundfile as sf
import streamlit as st

# Function to encode image as Base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"File '{image_path}' not found! Ensure the image is in the correct directory.")
        return None
    
import base64

# Function to load CSS and replace background image
def load_css_with_image(file_name, image_path):
    # Read the CSS file
    with open(file_name, "r") as css_file:
        css = css_file.read()
    
    # Convert image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()
    
    # Replace the placeholder with the base64 image
    css = css.replace("IMAGE_PLACEHOLDER", f"data:image/jpg;base64,{base64_image}")
    return f"<style>{css}</style>"

# Function to load predefined text files
def load_text_from_file(file_name):
    with open(file_name, "r") as file:
        return file.read()
    
