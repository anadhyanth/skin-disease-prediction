# Skin Disease Classification Using Deep Learning

## Project Overview

This project focuses on the classification of skin diseases using Deep Learning and Computer Vision techniques. The system analyzes dermatological images and predicts the corresponding skin disease category based on visual patterns and features.

The project utilizes Convolutional Neural Networks (CNNs) to automatically learn disease-specific characteristics from images and perform accurate classification.

---

## Objectives

* Detect and classify skin diseases from images.
* Automate dermatological image analysis.
* Apply Deep Learning techniques in healthcare.
* Support early disease identification and diagnosis.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn

---

## Dataset Description

The dataset contains images belonging to different skin disease categories.

Example classes:

* Acne
* Eczema
* Psoriasis
* Melanoma
* Ringworm
* Healthy Skin

The model can be adapted for binary or multi-class classification depending on the dataset.

---

## Data Preprocessing

The following preprocessing techniques are applied:

* Image Loading
* Image Resizing
* Pixel Normalization
* Label Encoding
* Data Augmentation
* Train-Test Split

---

## Project Workflow

### 1. Data Collection

Skin disease images are collected and organized into class-wise folders.

### 2. Image Preprocessing

* Resize images
* Normalize pixel values
* Convert images into tensors

### 3. Data Augmentation

* Rotation
* Zoom
* Horizontal Flip
* Width Shift
* Height Shift

### 4. CNN Model Development

A Convolutional Neural Network is built from scratch to learn visual features from skin disease images.

### 5. Model Training

The CNN model is trained using the prepared dataset.

### 6. Model Evaluation

Performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### 7. Disease Prediction

The trained model predicts skin disease categories for unseen images.

---

## Model Architecture

The CNN architecture consists of:

* Convolution Layers
* Batch Normalization Layers
* Max Pooling Layers
* Dropout Layers
* Dense Layers
* Output Layer

---

## Applications

* Dermatology Assistance
* Skin Disease Screening
* Medical Image Analysis
* Healthcare AI Systems
* Clinical Decision Support

---

## Installation

```bash
git clone https://github.com/your-username/skin-disease-classification.git

cd skin-disease-classification

pip install -r requirements.txt
```

---

## Running the Project

```bash
python main.py
```

---

## Future Enhancements

* Transfer Learning using ResNet50
* Mobile Deployment
* Real-Time Skin Disease Detection
* Explainable AI Visualization
* Cloud-Based Healthcare Application

---

## Author

B. Anadhyanth

B.Tech – Artificial Intelligence and Machine Learning

---

## Conclusion

This project demonstrates the use of Deep Learning and Computer Vision techniques for automated skin disease classification. The system can assist healthcare professionals by providing faster and more consistent image-based disease identification.
