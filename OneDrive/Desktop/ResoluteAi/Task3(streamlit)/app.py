import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to load and resize the image
def load_and_resize_image(image, width=1080, height=720):
    img = Image.open(image)
    img_resized = img.resize((width, height))
    return np.array(img_resized)

# Function to convert image to grayscale
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Function to apply blur effect
def apply_blur(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

# Function to apply edge detection
def detect_edges(image, threshold1=100, threshold2=150):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray_image, threshold1, threshold2)

# Streamlit UI
st.title("ResoluteAI Software")
st.title("TASK 3: IMAGE PROCESSING-Streamlit")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load and resize the image
    image = load_and_resize_image(uploaded_file)
    st.image(image, caption="Resized Image (1080x720 pixels)", use_column_width=True)
    
    # Buttons to apply effects
    if st.button("Gray"):
        gray_image = convert_to_gray(image)
        st.image(gray_image, caption="Grayscale Image", use_column_width=True)
    
    if st.button("Blur"):
        blurred_image = apply_blur(image)
        st.image(blurred_image, caption="Blurred Image", use_column_width=True)
    
    if st.button("Edge"):
        edge_image = detect_edges(image)
        st.image(edge_image, caption="Edge-detected Image", use_column_width=True)
