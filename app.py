import streamlit as st
import numpy as np
from PIL import Image
# 1. UPGRADED: Importing ResNet50 instead of MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Universal Object Recognizer", page_icon="🌍")
st.title("Universal Object Recognizer 🧠 (ResNet50 Upgrade)")
st.write("Upload an image of almost anything. This app now uses the heavier ResNet50 model for better accuracy!")

# 2. UPGRADED: Loading the heavier ResNet50 brain
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image to 224x224 RGB
    image = image.resize((224, 224))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply ResNet50's specific math processing
    img_array = preprocess_input(img_array)
    
    if st.button('Predict Object'):
        with st.spinner('Analyzing with ResNet50...'):
            prediction = model.predict(img_array)
            results = decode_predictions(prediction, top=3)[0]
            
            st.success(f"**Top Prediction: {results[0][1].replace('_', ' ').title()}**")
            st.info(f"Confidence: {results[0][2] * 100:.2f}%")
            
            st.write("Other possibilities:")
            for i in range(1, 3):
                st.write(f"- {results[i][1].replace('_', ' ').title()}: {results[i][2] * 100:.2f}%")
