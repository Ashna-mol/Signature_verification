import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pandas as pd

# Disable ONEDNN optimizations and suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Register the missing function
@tf.keras.utils.register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

# Define contrastive loss function
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

# Load Siamese Model with registered functions
@st.cache_resource
def load_siamese_model():
    return load_model("model\siamese_model .h5", custom_objects={'contrastive_loss': contrastive_loss, 'euclidean_distance': euclidean_distance})

# Image preprocessing function
def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (100, 100))  # Resize for model input
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=[0, -1])  # Add batch & channel dim
    return image

# Streamlit App
st.title("Signature Verification Using Siamese Network")

# Upload images
uploaded_file_a = st.file_uploader("Upload Signature A", type=["png", "jpg", "jpeg"])
uploaded_file_b = st.file_uploader("Upload Signature B", type=["png", "jpg", "jpeg"])

if uploaded_file_a and uploaded_file_b:
    if st.button("Verify Signatures"):
        model = load_siamese_model()
        img1 = preprocess_image(uploaded_file_a)
        img2 = preprocess_image(uploaded_file_b)

        # Get similarity score
        prediction = model.predict([img1, img2])
        similarity_score = prediction[0][0]
        threshold = 0.5  # Define a threshold for similarity
        result = "Match" if similarity_score < threshold else "No Match"

        # Display results
        st.image([uploaded_file_a, uploaded_file_b], caption=["Signature A", "Signature B"], width=200)
        st.write(f"*Similarity Score:* {similarity_score:.4f}")
        st.write(f"*Verification Result:* {result}")

        # Display DataFrame with results
        df = pd.DataFrame({
            "Signature A": [uploaded_file_a.name],
            "Signature B": [uploaded_file_b.name],
            "Similarity Score": [similarity_score],
            "Result": [result]
        })
        st.dataframe(df)