# 🚦 Traffic Sign Recognition with Explainable Deep Learning

A full-stack deep learning project that combines robust classification of German traffic signs using Convolutional Neural Networks (CNNs) with interpretability powered by Grad-CAM. The system is production-ready, featuring a Streamlit web app interface and modular training pipeline.

---

## 📌 Project Objective

> **Goal:** Build a scalable and interpretable computer vision system for real-time traffic sign recognition using CNNs trained on the GTSRB dataset, and empower end-users with **visual explanations** of model decisions through Grad-CAM.

---

## 🧠 Model Architecture

- **Model:** Custom CNN with 3 convolutional layers + ReLU + MaxPool + Dropout + FC layers
- **Input size:** 32×32 RGB images
- **Output:** 43 softmax-activated class probabilities
- **Training Acc:** ~99%  
- **Validation Acc:** ~97%

The model is trained from scratch and optimized using the Adam optimizer with categorical cross-entropy loss.

---

## 🔍 Explainability with Grad-CAM

This project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight spatial regions in the input that influence the model’s decision. Grad-CAM helps diagnose:
- Overfitting or spurious correlations
- Model reliance on texture vs shape
- Failure cases and class confusion

Grad-CAM overlays are available both in-batch (`notebooks/2_gradcam_analysis.ipynb`) and live within the Streamlit UI.


---

## 🔗 Dataset

- **Source:** [GTSRB – German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Classes:** 43 traffic sign categories
- **Total Images:** 39,209 (Train) + 12,630 (Test)
- **Preprocessing:** Resize to 32×32, Normalize to [-1, 1]

⚠️ **Note:** The dataset is not included in this repository due to size.


---

**🔗 Try the live app here:** [Streamlit App](https://traffic-sign-recognition-27kcgm5fysldhkfbnq4g6x.streamlit.app/)
