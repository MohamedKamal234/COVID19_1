import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(page_title="Covid-19 Detection", layout="centered")
st.title("🩺 Covid-19 Detection System (CNN)")

# 2. Sidebar - Team Members
st.sidebar.title("Powered by:")
st.sidebar.markdown("""
* Omar Mohamed Elsayed Ahmed Wahdan
* Abdelrahman Mohamed Talaat Sokar
* Mohamed Kamal Ahmed Elashmawy
* Diaa Essam Fathy
* Hossam Amr Mohamed
* Ali Basyoni Gad
""")

# 3. Load the Model
@st.cache_resource
def load_my_model():
    # Loading the .keras model saved from your notebook
    model_path = os.path.join(os.getcwd(), 'final_model.keras')
    model = tf.keras.models.load_model(model_path)
    return model

try:
    model = load_my_model()
    st.sidebar.success("Model loaded successfully ✅")
except Exception as e:
    st.sidebar.error("Error: my_model.keras not found in the directory.")
    st.stop()

# 4. Image Uploading
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.info("Classifying... please wait.")
    
    # Image Preprocessing (128x128)
    img = image.resize((128, 128))
    img = img.convert('RGB') 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Prediction
    predictions = model.predict(img_array)
    class_names = ['Normal', 'Covid-19']
    result_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    # 6. Display Results
    st.subheader(f"Result: {class_names[result_index]}")
    st.write(f"Confidence Level: {confidence:.2f}%")
    
    if class_names[result_index] == 'Covid-19':
        st.error("⚠️ Positive: Covid-19 Detected")
    else:
        st.success("✅ Negative: Normal Case")