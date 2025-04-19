# Traffic Sign Recognition with Explainable Deep Learning

A full-stack deep learning project that combines robust classification of German traffic signs using a custom CNN model, with integrated explainability via Grad-CAM. Built for real-time inference with a modular architecture and a deployed Streamlit interface.

---

## Project Overview

This project delivers a scalable and interpretable computer vision pipeline for recognizing traffic signs in real time. It leverages the GTSRB dataset and prioritizes both model accuracy and transparency in decision-making.

---

## Model Architecture

- Custom CNN architecture: 3 convolutional layers + ReLU + MaxPool + Dropout + Fully Connected layers  
- Input: 32×32 RGB images  
- Output: 43 softmax class probabilities  
- Optimizer: Adam  
- Loss: Categorical cross-entropy  
- Performance: ~99% train accuracy, ~97% validation accuracy  

---

## Interpretability with Grad-CAM

To ensure model accountability, the system integrates Grad-CAM (Gradient-weighted Class Activation Mapping). This allows users to visualize which regions in the image contributed most to the model's prediction.

Grad-CAM is implemented both:
- Offline: via `notebooks/2_gradcam_analysis.ipynb`
- Online: directly within the Streamlit interface

---

## Dataset

- Source: [GTSRB – German Traffic Sign Recognition Benchmark (Kaggle)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- 43 classes, ~51,000 total images
- Preprocessing:
  - Resized to 32×32 pixels
  - Normalized to [-1, 1]

*Note: Dataset not included due to size restrictions.*

---

## Deployment

- Web UI built using Streamlit
- Supports real-time image upload and classification with interpretability

**Live Demo:** [https://traffic-sign-recognition-27kcgm5fysldhkfbnq4g6x.streamlit.app/](https://traffic-sign-recognition-27kcgm5fysldhkfbnq4g6x.streamlit.app/)
