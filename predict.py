import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("crop_disease_model.h5")

# Load class indices mapping
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping {index: class_name}
idx_to_class = {v: k for k, v in class_indices.items()}

def format_label(label):
    """Convert 'Tomato___Late_blight' → 'Tomato | Late Blight' """
    parts = label.split("___")
    if len(parts) == 2:
        plant, disease = parts
        disease = disease.replace("_", " ").title()
        return f"Plant: {plant} | Disease: {disease}"
    else:
        return label.replace("_", " ").title()

def predict_crop(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    confidence = np.max(predictions)

    class_label = idx_to_class[predicted_class_idx]
    readable_label = format_label(class_label)

    print(f"{readable_label} | Confidence: {confidence*100:.2f}%")

# Example usage
predict_crop(r"E:\FRIDAY\SIDDARTH_S\IMP\PROJECTS\CROP DISEASE DETECTION\dataset\test\potato_healthy_114.JPG")
