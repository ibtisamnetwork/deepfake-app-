import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
    model = models.shufflenet_v2_x1_0(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("best_shufflenet.pth", map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# ====== IMAGE TRANSFORMS ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== HEADER ======
st.markdown("<h1 style='text-align: center;'>🕵️‍♂️ DeepFake Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image and let AI detect if it's <strong>Real</strong> or <strong>Fake</strong>.</p>", unsafe_allow_html=True)
st.markdown("---")

# ====== FILE UPLOAD ======
uploaded_file = st.file_uploader("📤 Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='🖼 Uploaded Image', use_column_width=True)

    # ====== PREPROCESS ======
    img_tensor = transform(image).unsqueeze(0).to("cpu")

    # ====== PREDICT ======
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_names = ['🟥 Fake', '🟩 Real']
        pred_class = class_names[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()] * 100

    # ====== RESULT DISPLAY ======
    st.markdown("---")
    st.markdown(
        f"""
        <div style='
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        '>
            <h3 style='margin-bottom: 10px;'>🧠 Prediction: <span style='color: #4B8BBE'>{pred_class}</span></h3>
            <p style='font-size: 18px;'>Confidence: <strong>{confidence:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("🔍 <i>This result is based on the uploaded image and may not be perfect. Always verify with additional tools.</i>", unsafe_allow_html=True)


