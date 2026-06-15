import streamlit as st
import tensorflow as tf
import numpy as np
import json
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

# Load recommendations from JSON file
with open("recommendations.json", "r", encoding="utf-8") as f:
    recommendations = json.load(f)

# Sidebar
# Sidebar
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("📂 Select Page", ["HOME", "DISEASE RECOGNITION"])

# Divider
st.sidebar.markdown("---")

# About section
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
👨‍💻 **Developer:** Ritesh Khurana  
⚡ **Powered by:** TensorFlow & Streamlit  

🛠️ **Tech Stack:**  
- Python  
- TensorFlow  
- Keras  
- Streamlit  
- Pillow  
- NumPy  

📂 **Model File:** `trained_plant_disease_model.keras`  
📘 **Recommendations File:** `recommendations.json`  
""")

# Divider
st.sidebar.markdown("---")

# Supported crops section
st.sidebar.markdown("### 🌾 Supported Crops")
st.sidebar.markdown("""
- 🍎 Apple  
- 🌽 Corn  
- 🍅 Tomato  
- 🥔 Potato  
- 🍇 Grape  
- 🍑 Peach  
- 🌶️ Bell Pepper  
""")

# Divider
st.sidebar.markdown("---")

# Links & Contact
st.sidebar.markdown("🐙 [GitHub Repo](ttps://github.com/Ritesh-Khurana/Plant-Disease-Detection)")
st.sidebar.markdown("📧 Contact: riteshkhurana97@.com")

st.sidebar.markdown("⚡ **Powered by:** TensorFlow & Streamlit ")

# import Image from pillow to open images
from PIL import Image
img = Image.open("Diseases.png")

# display image using streamlit
# width is used to set the width of an image
st.image(img)

#Main Page
if app_mode == "HOME":
    st.title("🌱 Plant Disease Detection System")
    st.subheader("For Sustainable Agriculture")

    # Intro
    st.markdown("""
    Welcome! This app uses a deep learning model to identify plant diseases from leaf images.  
    Upload a photo of a leaf and get instant predictions **plus AI-powered recommendations** 
    to support better crop management.
    """)

    # Quick Start
    st.info("📘 Quick Start: 1️⃣ Upload leaf → 2️⃣ Click Predict → 3️⃣ Get AI tips 🌿")

    # Supported crops
    st.markdown("### 🌾 Supported Crops & Diseases")
    st.markdown("""
    - 🍎 **Apple** → Apple Scab, Black Rot, Cedar Apple Rust, Healthy  
    - 🌽 **Corn** → Common Rust, Northern Leaf Blight, Healthy  
    - 🍅 **Tomato** → Early Blight, Late Blight, Leaf Mold, Healthy  
    - 🥔 **Potato** → Early Blight, Late Blight, Healthy  
    - 🍇 **Grape** → Black Rot, Healthy  
    - 🍑 **Peach** → Bacterial Spot, Healthy  
    - 🌶️ **Bell Pepper** → Bacterial Spot, Healthy  
    - 🌱 **General** → Healthy plant detection
    """)

    # How to use
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. 📷 Go to **Disease Recognition** page  
    2. 🖼️ Upload a clear leaf image (JPG/PNG)  
    3. 🖱️ Click **Predict** to see classification  
    4. 💡 Get instant **AI-powered recommendations** for treatment or prevention  
    """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<h5 style='text-align: center;'>👨‍💻 Developed by <b>Ritesh Khurana</b> | ⚡ Powered by TensorFlow & Streamlit 🌱</h5>",
        unsafe_allow_html=True
    )

    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:", type=["jpg","jpeg","png"])
    
    if test_image is not None:
        # Show uploaded image
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")

            # Save uploaded file temporarily
            with open("temp.jpg", "wb") as f:
                f.write(test_image.getbuffer())

            # Run prediction
            result_index = model_prediction("temp.jpg")

            # Reading Labels
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            predicted_label = class_name[result_index]
            st.success(f"✅ Model is Predicting it's a {predicted_label}")

             # Show recommendation
            if predicted_label in recommendations:
             st.info(f"🌿 Recommendation: {recommendations[predicted_label]}")
            else:
             st.info("🌿 Recommendation: General care — monitor regularly and maintain soil health.")
