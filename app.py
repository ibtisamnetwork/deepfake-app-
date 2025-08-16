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

# ====== CUSTOM CSS ======
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff;
        padding: 2rem 1rem 4rem 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Style uploader box */
    div[data-testid="fileUploaderDropzone"] {
        background-color: rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 2px dashed #dcd6f7;
        color: white;
        font-weight: 600;
    }
    /* Uploaded image */
    img {
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 20px;
    }
    /* Result box */
    .result-box {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 1.5rem;
        max-width: 450px;
        margin: 1.5rem auto;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        text-align: center;
    }
    /* Prediction text colors */
    .pred-fake {
        color: #f44336; /* red */
        font-weight: 700;
        font-size: 1.5rem;
    }
    .pred-real {
        color: #4caf50; /* green */
        font-weight: 700;
        font-size: 1.5rem;
    }
    /* Header styles */
    h1, p {
        text-align: center;
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

# ====== IMAGE TRANSFORMS ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== HEADER ======
st.markdown("<h1>🕵️‍♂️ DeepFake Detector</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload an image and let AI detect if it's <strong>Real</strong> or <strong>Fake</strong>.</p>", unsafe_allow_html=True)
st.markdown("---")

# ====== FILE UPLOAD ======
uploaded_file = st.file_uploader("📤 Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='🖼 Uploaded Image', use_column_width=False, width=350)

    # ====== PREPROCESS ======
    img_tensor = transform(image).unsqueeze(0).to("cpu")

    # ====== PREDICT ======
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_names = ['Fake', 'Real']
        pred_class = class_names[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()] * 100

    # ====== DISPLAY RESULT ======
    color_class = "pred-real" if pred_class == "Real" else "pred-fake"

    st.markdown(
        f"""
        <div class="result-box">
            <h3>🧠 Prediction: <span class="{color_class}">{pred_class}</span></h3>
            <p>Confidence: <strong>{confidence:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("🔍 <i>This result is based on the uploaded image and may not be perfect. Always verify with additional tools.</i>", unsafe_allow_html=True)
