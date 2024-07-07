import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('raw_models/inceptionv3_tf.h5')
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return None

model = load_model()

def predict_nutrition(image):
    if image is None:
        raise ValueError("No image provided")
    
    logger.info(f"Starting prediction for image of size {image.size}")
    
    try:
        img = image.resize((224, 224))
        img = img.convert('RGB')  # Convert image to RGB
        img_array = np.array(img)
        
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Invalid image shape: {img_array.shape}")
        
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        logger.info("Prediction successful")
        
        
        nutrition_data = {
            'calories': float(prediction[0][0]),
            'protein': float(prediction[0][1]),
            'carbohydrates': float(prediction[0][2]),
            'fat': float(prediction[0][3])
        }
        
        return nutrition_data
    
    except Exception as e:
        logger.error(f"Error in predict_nutrition: {str(e)}", exc_info=True)
        return None

st.title("Diet Vision")
st.markdown("***A Diet Vision For a Healthier Tomorrow.***")

uploaded_file = st.file_uploader("Choose a meal photo", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Meal Photo", use_column_width=True)
        
        nutrition_data = predict_nutrition(image)
        
        if nutrition_data is not None:
            st.header("Nutrition Information")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Mass", f"{nutrition_data.get('mass', 0):.0f} g")
            col2.metric("Calories", f"{nutrition_data.get('calories', 0):.1f} c")
            col3.metric("Protein", f"{nutrition_data.get('protein', 0):.1f} g")
            col4.metric("Carbs", f"{abs(nutrition_data.get('carbohydrates', 0)):.1f} g")
            col5.metric("Fat", f"{nutrition_data.get('fat', 0):.1f} g")

            st.header("Daily Nutrition Goals")
            calories_progress = nutrition_data.get('calories', 0) / 2000  # Assuming 2000 kcal daily goal
            st.progress(calories_progress)
            st.text(f"{calories_progress*100:.1f}% of daily calorie goal")

            st.header("Nutrients")
            for nutrient, amount in nutrition_data.items():
                st.text(f"{nutrient.capitalize()}: {abs(amount):.1f}")

            st.header("Adjust Consumed Amount")
            consumed_percentage = st.slider("Percentage of meal consumed", 0, 100, 100)
            
            # Update nutrition
            adjusted_nutrition = {k: v * consumed_percentage / 100 for k, v in nutrition_data.items()}
            
            st.header("Adjusted Nutrition")
            col1, col2, col3 = st.columns(3)
            col1.metric("Adjusted Calories", f"{adjusted_nutrition.get('calories', 0):.0f} cal")
            col2.metric("Adjusted Protein", f"{adjusted_nutrition.get('protein', 0):.1f} g")
            col3.metric("Adjusted Carbs", f"{abs(adjusted_nutrition.get('carbohydrates', 0)):.1f} g")
        else:
            st.error("Failed to predict nutrition data. Please try again with a different image.")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error in main app flow: {str(e)}", exc_info=True)
else:
    st.info("Please upload a meal photo to start tracking your nutrition.")