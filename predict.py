import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the multi-class model
model_path = "model/skin_cancer_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = None

# 9 Classes matching the dataset folder structure
classes = [
    'actinic_keratosis',
    'basal_cell_carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented_benign_keratosis',
    'seborrheic_keratosis',
    'squamous_cell_carcinoma',
    'vascular_lesion'
]

# Define which classes are considered "Cancer"
cancer_classes = [
    'melanoma',
    'basal_cell_carcinoma',
    'squamous_cell_carcinoma'
]

def predict_skin(img_path):
    if model is None:
        return "Model not found", "Unknown", 0.0

    # Load and process image
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    pred = model.predict(img_array)
    
    # Get highest confidence and class index
    confidence = np.max(pred) * 100
    class_index = np.argmax(pred)
    class_name = classes[class_index]

    # Determine if it's "Cancer Detected" or "No Cancer"
    if class_name in cancer_classes:
        result = "Cancer Detected"
    else:
        result = "No Cancer"

    return result, class_name, confidence
