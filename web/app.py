import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import datetime
import requests

# Load the trained model
model = tf.keras.models.load_model('models/inceptionv3_model.h5')

# Function to get data from API
def get_meal_data(image):
    url = 'http://34.203.252.91/'
    # Read the image file
    with io.BytesIO() as img_file:
        image.save(img_file, format='PNG')
        img_file.seek(0)
        files = {'image': img_file}
        response = requests.post(url, files=files, verify=False)  # Set verify=True if you have a valid SSL certificate
    
    if response.status_code == 200:
        return response.json()
    else:
        return {}

# Function to predict nutrition from image
def predict_nutrition(image):
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Use the actual API to get the nutrition data
    nutrition_data = get_meal_data(image)
    
    return nutrition_data

# Set page config
st.title("Diet Vision")
st.markdown("***A Diet Vision For a Healthier Tomorrow.***")

# File uploader for meal photo
uploaded_file = st.file_uploader("Choose a meal photo", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Meal Photo", use_column_width=True)
    
    # Predict nutrition from the image
    nutrition_data = predict_nutrition(image)
    
    if nutrition_data:
        # Display nutrition information
        st.header("Nutrition Information")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mass", f"{nutrition_data.get('mass', 0):.0f} g")
        col2.metric("Calories", f"{nutrition_data.get('calories', 0):.1f} c")
        col3.metric("Protein", f"{nutrition_data.get('protein', 0):.1f} g")
        col4.metric("Carbs", f"{abs(nutrition_data.get('carbohydrates', 0)):.1f} g")
        col5.metric("Fat", f"{nutrition_data.get('fat', 0):.1f} g")
        
        # Daily nutrition goals progress
        st.header("Daily Nutrition Goals")
        calories_progress = nutrition_data.get('calories', 0) / 2000  # Assuming 2000 kcal daily goal
        st.progress(calories_progress)
        st.text(f"{calories_progress*100:.1f}% of daily calorie goal")
        
        # Nutrients list
        st.header("Nutrients")
        for nutrient, amount in nutrition_data.items():
            st.text(f"{nutrient.capitalize()}: {abs(amount):.1f}")

        
        # Interactive percentage bar
        st.header("Adjust Consumed Amount")
        consumed_percentage = st.slider("Percentage of meal consumed", 0, 100, 100)
        
        # Update nutrition based on consumed percentage
        adjusted_nutrition = {k: v * consumed_percentage / 100 for k, v in nutrition_data.items()}
        
        st.header("Adjusted Nutrition")
        col1, col2, col3 = st.columns(3)
        col1.metric("Adjusted Calories", f"{adjusted_nutrition.get('calories', 0):.0f} cal")
        col2.metric("Adjusted Protein", f"{adjusted_nutrition.get('protein', 0):.1f} g")
        col3.metric("Adjusted Carbs", f"{abs(adjusted_nutrition.get('carbohydrates', 0)):.1f} g")

    else:
        st.error("Failed to retrieve nutrition data from the API.")
else:
    st.info("Please upload a meal photo to start tracking your nutrition.")