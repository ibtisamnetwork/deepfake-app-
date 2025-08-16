import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import time

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# ====== CUSTOM CSS ======
st.markdown("""
<style>
    .stApp {
        background-color: #f0f7ff;
        color: #222222;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 2rem 1rem 4rem 1rem;
    }
    .header {
        background: linear-gradient(90deg, #4B8BBE, #306998);
        padding: 2rem 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 15px rgba(75,139,190,0.4);
        margin-bottom: 1.5rem;
    }
    div[data-testid="fileUploaderDropzone"] {
        background: #61a0af;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.5rem;
        border: none;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(97,160,175,0.5);
        transition: background-color 0.3s ease;
    }
    div[data-testid="fileUploaderDropzone"]:hover {
        background: #468a96;
        box-shadow: 0 6px 20px rgba(70,138,150,0.7);
    }
    img {
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        max-width: 350px;
        height: auto;
        display: block;
        margin: 0 auto 25px auto;
    }
    .result-box {
        background-color: #e9f0f7;
        border-radius: 15px;
        padding: 2rem;
        max-width: 450px;
        margin: 1.5rem auto;
        box-shadow: 0 6px 24px rgba(48,105,152,0.25);
        text-align: center;
        color: #222222;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .pred-fake {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        box-shadow: 0 3px 10px rgba(211,47,47,0.3);
        display: inline-block;
        font-weight: 700;
        font-size: 1.4rem;
        margin-left: 10px;
    }
    .pred-real {
        color: #388e3c;
        background-color: #e8f5e9;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        box-shadow: 0 3px 10px rgba(56,142,60,0.3);
        display: inline-block;
        font-weight: 700;
        font-size: 1.4rem;
        margin-left: 10px;
    }
    .tagline {
        text-align: center;
        color: #555555;
        font-style: italic;
        margin-top: -10px;
        margin-bottom: 25px;
        font-size: 1rem;
    }
    .footer {
        font-size: 0.85rem;
        text-align: center;
        margin-top: 3rem;
        color: #555555;
        font-style: italic;
    }
    .reset-button {
        text-align: center;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== HEADER ======
st.markdown("""
<div class="header">
    <h1>🕵️‍♂️ DeepFake Detector</h1>
    <p>Upload an image and let AI detect if it's <strong>Real</strong> or <strong>Fake</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ====== SESSION STATE SETUP ======
if "pred_class" not in st.session_state:
    st.session_state.pred_class = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "image" not in st.session_state:
    st.session_state.image = None

# ====== FILE UPLOADER ======
uploaded_file = st.file_uploader("📤 Choose an image file", type=["jpg", "jpeg", "png"], key="file_uploader")

st.markdown('<p class="tagline">Upload a face image to detect deepfakes — stay aware!</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")
    st.image(st.session_state.image, caption="🖼 Uploaded Image")

# ====== ANALYZE BUTTON ======
if st.session_state.image is not None:
    analyze_clicked = st.button("🧠 Analyze Image")

    if analyze_clicked:
        with st.spinner("Processing..."):
            progress_bar = st.progress(0)
            img_tensor = transform(st.session_state.image).unsqueeze(0).to("cpu")
            
            # Simulate progress
            for percent_complete in range(0, 101, 20):
                time.sleep(0.2)
                progress_bar.progress(percent_complete)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                class_names = ['Fake', 'Real']
                st.session_state.pred_class = class_names[predicted.item()]
                st.session_state.confidence = torch.softmax(outputs, dim=1)[0][predicted.item()] * 100
            
            progress_bar.empty()

# ====== DISPLAY RESULT ======
if st.session_state.pred_class is not None:
    color_class = "pred-real" if st.session_state.pred_class == "Real" else "pred-fake"
    st.markdown(
        f"""
        <div class="result-box">
            <span>🧠 Prediction:</span> <span class="{color_class}">{st.session_state.pred_class}</span>
            <p>Confidence: <strong>{st.session_state.confidence:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ====== RESET BUTTON ======
def reset():
    st.session_state.pred_class = None
    st.session_state.confidence = None
    st.session_state.image = None
    st.session_state.file_uploader = None
    st.experimental_rerun()

st.markdown('<div class="reset-button">', unsafe_allow_html=True)
if st.button("🔄 Reset"):
    reset()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div class='footer'>🔍 This result is based on the uploaded image and may not be perfect. Always verify with additional tools.</div>", unsafe_allow_html=True)
