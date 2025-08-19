# 🌱 Crop Disease Detection using Deep Learning

## 📌 Overview

This project is a **deep learning-based image classification system** that detects crop diseases from leaf images.
It uses a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras to classify plant leaves into **healthy** or various **disease categories**.

The goal is to assist farmers and researchers in **early detection of plant diseases**, which can improve yield and reduce crop loss.

---

## 🚀 Features

✅ Train a CNN model on plant disease datasets
✅ Supports multiple crop and disease classes
✅ Real-time prediction on new leaf images
✅ Data augmentation for robust training
✅ Model checkpointing & early stopping
✅ Easy-to-use prediction script (`predict.py`)

---

## 🗂️ Dataset

The model is trained on the **PlantVillage dataset** (or any dataset structured in the following format):

```
dataset/
│── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

📥 Download PlantVillage dataset:

* [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/crop-disease-detection.git
   cd crop-disease-detection
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Training

Run the training script:

```bash
python train.py
```

This will:

* Train the CNN model on the dataset
* Save the best model as `crop_disease_model.h5`
* Save class labels in `class_indices.json`

---

## 🔎 Prediction

Use `predict.py` to classify a leaf image:

```bash
python predict.py --image "path/to/leaf.jpg"
```

### Example Output:

```
✅ Prediction: Potato - Early Blight
Confidence: 95.3%
```

---

## 📊 Model Architecture

* **Conv2D + MaxPooling** layers (feature extraction)
* **Flatten** + **Dense** layers (classification)
* **Dropout** (to prevent overfitting)
* **Softmax output layer**

---

## 📈 Results

* Accuracy: \~95% (on validation dataset)
* Robust against rotation, zoom, and flips (due to augmentation)

---

## 🔮 Future Improvements

* Add more crops & diseases
* Deploy as a **web app** or **mobile app**
* Integrate with IoT devices for smart farming

---

DATASET-https://www.kaggle.com/datasets/alyeko/potato-tomato-dataset?utm_source=chatgpt.com
