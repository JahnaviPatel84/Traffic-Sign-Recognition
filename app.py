import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import streamlit as st

# ==== Setup ====
st.set_page_config(page_title="🚦 Traffic Sign Recognition", layout="wide")
st.title("🚦 Traffic Sign Recognition with Grad-CAM")
st.markdown("Upload a traffic sign image to see the predicted class and attention heatmap.")

# ==== Class Labels ====
CLASS_ID_TO_NAME = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles > 3.5 tons',
    11: 'Right-of-way at next intersection', 12: 'Priority road', 13: 'Yield', 14: 'Stop',
    15: 'No vehicles', 16: 'Vehicles > 3.5 tons prohibited', 17: 'No entry',
    18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all restrictions', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End of no passing for vehicles > 3.5 tons'
}

# ==== Model ====
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# ==== Grad-CAM ====
def generate_gradcam(model, image_tensor, class_idx, target_layer='conv3'):
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    layer = getattr(model, target_layer)
    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_backward_hook(backward_hook)

    image_tensor = image_tensor.unsqueeze(0)
    output = model(image_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    handle_f.remove()
    handle_b.remove()

    act = activations[0].squeeze().cpu().detach().numpy()
    grad = gradients[0].squeeze().cpu().detach().numpy()
    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (32, 32))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# ==== Load Model ====
model = TrafficSignCNN()
model.load_state_dict(torch.load("models/traffic_sign_cnn.pth", map_location="cpu"))
model.eval()

# ==== Image Upload ====
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### Uploaded Image")

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = transform(image)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        pred_class = output.argmax(dim=1).item()
        pred_confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        pred_name = CLASS_ID_TO_NAME[pred_class]

    st.success(f"**Prediction: {pred_name}**")
    st.markdown(f"**Confidence: {pred_confidence * 100:.2f}%**")

    # Grad-CAM
    cam = generate_gradcam(model, input_tensor, pred_class)
    image_np = input_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 0.5 + 0.5).clip(0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(image_np * 255), 0.6, heatmap, 0.4, 0)

    # Show results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_np, caption="Uploaded Image", width=250)
    with col2:
        st.image(overlay[:, :, ::-1], caption="Grad-CAM Heatmap", width=250)

    # Optional: save image
    cv2.imwrite("gradcam_output.png", overlay[:, :, ::-1])
    with open("gradcam_output.png", "rb") as f:
        st.download_button("Download Grad-CAM Image", f, file_name="gradcam_output.png")

    # Feedback
    st.markdown("---")
    st.markdown("### 🤔 Was the prediction correct?")
    feedback = st.radio("Select an option:", ["Yes", "No", "Not sure"])
    if feedback:
        st.info("Thanks for your feedback!")
