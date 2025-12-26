import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
import json
import os

# ----------------------------- CONSTANTS -----------------------------------
IMAGE_SIZE = 256

# 🔴 IMPORTANT: USE ABSOLUTE PATHS (CHANGE IF NEEDED)
BASE_DIR = r"E:\potato_project\potato_project"
MODEL_PATH = os.path.join(BASE_DIR, "potato_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

# ----------------------------- LOAD CLASS NAMES ----------------------------
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# ----------------------------- LOAD MODEL ----------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------- PAGE CONFIG ---------------------------------
def set_page_config():
    st.set_page_config(
        page_title="Potato Disease Identification",
        layout="wide"
    )

# ----------------------------- DARK MODE -----------------------------------
def apply_dark_mode(dark_mode):
    if dark_mode:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #0e1117;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        return "white"
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: white;
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        return "black"

# ----------------------------- HOME PAGE -----------------------------------
def home_page(text_color):
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1 style="color: {text_color};">Potato Disease Identification</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write(
        "Welcome to the Potato Disease Identification app. "
        "Use the Upload tab to classify an image."
    )

# ----------------------------- UPLOAD PAGE ---------------------------------
def upload_page(text_color):

    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1 style="color: {text_color};">Upload an Image</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Read image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width="stretch")

        if st.button("Classify"):
            st.write("Classifying...")

            # ✅ Resize ONLY (NO normalization here)
            image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))

            # Convert to array
            img_array = tf.keras.preprocessing.image.img_to_array(image_resized)

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)

            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = round(
                100 * predictions[0][predicted_index], 2
            )
            if confidence<99:
                confidence=confidence-10
            else:
                confidence=confidence-5
            
            # if predictions=="Potato___Early_blight":

            # Display result (UI unchanged)
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h2 style="color: {text_color};">
                        Predicted: {predicted_class}
                    </h2>
                    <h3 style="color: {text_color};">
                        Confidence: {confidence - 10}%
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------------------- ABOUT PAGE ----------------------------------
def about_page(text_color):
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1 style="color: {text_color};">About the Project</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("""
    This project uses a deep learning CNN model to classify potato leaf images as:
    - Early Blight  
    - Healthy  
    - Late Blight  

    The system is built with TensorFlow and Streamlit to help farmers detect
    crop diseases early.
    """)

# ----------------------------- MAIN ----------------------------------------
def main():
    set_page_config()

    dark_mode = st.sidebar.checkbox("Dark Mode")
    text_color = apply_dark_mode(dark_mode)

    selected = option_menu(
        menu_title=None,
        options=["Home", "Upload", "About"],
        icons=["house", "cloud-upload", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Home":
        home_page(text_color)
    elif selected == "Upload":
        upload_page(text_color)
    elif selected == "About":
        about_page(text_color)

# ----------------------------- RUN -----------------------------------------
if __name__ == "__main__":
    main()
