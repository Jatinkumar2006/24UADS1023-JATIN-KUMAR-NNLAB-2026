import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np

# -----------------------------------
# 🎯 PAGE CONFIG (UI SETUP)
# -----------------------------------
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🫁 Pneumonia Detection System</h1>
    <p style='text-align: center;'>Upload a chest X-ray image to detect Pneumonia</p>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# 🎯 DEVICE
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# 🎯 LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    model = resnet18(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Sequential(  # type: ignore
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )

    model.load_state_dict(torch.load("models/final_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------------
# 🎯 IMAGE TRANSFORM
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------------
# 🎯 FILE UPLOAD
# -----------------------------------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

# -----------------------------------
# 🎯 PREDICTION
# -----------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=400)
    # Convert image → tensor
    from typing import cast
    import torch

    img_tensor = cast(torch.Tensor, transform(image)) 
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    class_names = ["Normal", "Pneumonia"]
    result = class_names[int(pred.item())]
    confidence = confidence.item() * 100

    st.markdown("---")

    # -----------------------------------
    # 🎯 RESULT DISPLAY
    # -----------------------------------
    if result == "Pneumonia":
        st.error(f"⚠️ Pneumonia Detected\n\nConfidence: {confidence:.2f}%")
    else:
        st.success(f"✅ Normal\n\nConfidence: {confidence:.2f}%")

    # -----------------------------------
    # 🎯 PROBABILITY BAR
    # -----------------------------------
    st.subheader("Prediction Confidence")

    prob_values = probs.cpu().numpy()[0]

    st.write(f"Normal: {prob_values[0]*100:.2f}%")
    st.progress(float(prob_values[0]))

    st.write(f"Pneumonia: {prob_values[1]*100:.2f}%")
    st.progress(float(prob_values[1]))

# -----------------------------------
# 🎯 FOOTER
# -----------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 14px;'>
    Built using PyTorch + Streamlit | Transfer Learning (ResNet18)
    </p>
    """,
    unsafe_allow_html=True
)