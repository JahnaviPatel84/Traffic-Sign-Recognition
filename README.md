## Traffic Sign Recognition with Explainable Deep Learning

A production-ready deep learning pipeline for real-time traffic sign classification with integrated visual explainability. This project combines a custom-trained CNN with Grad-CAM to deliver interpretable, high-accuracy predictions, deployed via a clean and modular Streamlit interface.

---

### Project Overview

This project presents a modular, full-stack computer vision system capable of classifying German traffic signs in real time. It emphasizes both model performance and interpretability, making it suitable for deployment in environments where decision transparency is essential.

---

### Model Architecture

- Custom CNN with 3 convolutional layers, ReLU activations, max pooling, dropout, and fully connected layers
- Input: 32×32 RGB images  
- Output: Probability distribution across 43 traffic sign classes  
- Optimizer: Adam  
- Loss Function: Categorical Cross-Entropy  
- Performance:
  - ~99% training accuracy  
  - ~97% validation accuracy  
  - ~94% test accuracy on previously unseen data

The model was trained with data augmentation strategies such as random rotation and brightness jitter to improve generalization and reduce overfitting.

---

### Explainability with Grad-CAM

To ensure transparency in decision-making, the system incorporates Gradient-weighted Class Activation Mapping (Grad-CAM). This technique highlights the regions in the image that most influenced the model's predictions.

Implemented in two modes:
- Offline: Exploratory analysis in `notebooks/2_gradcam_analysis.ipynb`
- Online: Real-time Grad-CAM overlays directly in the deployed interface

---

### Dataset

- Source: German Traffic Sign Recognition Benchmark (GTSRB)
- Total Images: ~51,000
- Classes: 43
- Preprocessing:
  - Resized to 32×32 pixels
  - Normalized to the range [-1, 1]
  - Stratified train/validation/test split

Note: The dataset is not included in the repository due to size constraints.

---

### Deployment

- Frontend: Streamlit-based interactive web application  
- Functionality:
  - Image upload with real-time prediction
  - Class label and confidence score display
  - Grad-CAM heatmap rendering
  - Downloadable explanations
  - Optional user feedback collection
- Live Demo: [Streamlit App](https://traffic-sign-recognition-27kcgm5fysldhkfbnq4g6x.streamlit.app/)

---

### Timeline

Originally developed between January and May 2023. The project was packaged, documented, and publicly released in April 2025 to demonstrate an end-to-end computer vision workflow emphasizing transparency and explainability.

---

### Keywords

Deep Learning, Computer Vision, CNN, Grad-CAM, Explainable AI, Streamlit, Traffic Sign Classification, Real-Time Inference, Full-Stack ML, Model Interpretability, GTSRB
