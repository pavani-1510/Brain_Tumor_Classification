# Brain Tumor Classification of MRI Images

## Introduction
Brain tumors are one of the most life-threatening medical conditions. Early and accurate diagnosis is crucial for effective treatment. Manual diagnosis is time-consuming and prone to human error. This project aims to enhance diagnostic accuracy using AI-driven image classification to categorize brain tumor MRI images into four classes:
- Glioma
- Healthy
- Meningioma
- Pituitary

## Problem Statement
Distinguishing between various types of brain tumors is complex. Misclassification can lead to incorrect treatment and adverse patient outcomes. This project aims to develop a reliable, automated system for brain tumor classification using deep learning models.

## Dataset Overview
- **Description:** MRI dataset with four classes: Glioma, Healthy, Meningioma, and Pituitary.
- **Sources:**  
  - [Kaggle Dataset 1](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
  - [Kaggle Dataset 2](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri)
- **Data Distribution:**
  - Glioma: 2293 images
  - Healthy: 4000 images
  - Meningioma: 2757 images
  - Pituitary: 2386 images

## Preprocessing Techniques
Objective: Prepare MRI images for model training.
- **Resizing:** Standardized to (128, 128) pixels.
- **Normalization:** Pixel values scaled to [0, 1].
- **Data Augmentation:** Applied rotation, flipping, zooming, and brightness adjustment for better generalization.
- **Class Balancing:** Addressed class imbalance using oversampling.
- **Tools Used:** OpenCV, Keras ImageDataGenerator.

## Project Workflow
1. Data Collection and Exploration
2. Data Preprocessing
3. Model Selection and Architecture Design
4. Model Training and Validation
5. Performance Evaluation
6. Model Comparison and Selection

## Model Architecture
Implemented and compared multiple models:
- **VGG16**
- **ResNet50**
- **MobileNetV2**
- **Custom CNN Architecture**

## Model Training
- **Training Data:** Applied rescaling (rescale=1.0/255)
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross Entropy
- **Metrics:** Accuracy
- **Epochs:** 10
- **Validation Split:** 20% of data used for validation

## Results
- Achieved high accuracy and robustness across multiple models.
- Effective preprocessing and data augmentation contributed to improved performance.
- Comparison of models demonstrated superior performance of [Best Performing Model].

## Conclusion
- Successfully developed an automated brain tumor classification system using deep learning.
- The system aids in early and accurate diagnosis, supporting medical professionals.
- This approach enhances diagnostic efficiency and reduces human error in brain tumor detection.

## Author
- [R Pavani](https://www.linkedin.com/in/r-pavani/)
