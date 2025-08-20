import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ====== CONFIG ======
st.set_page_config(page_title="DeepFake Detector", page_icon="🕵️‍♂️", layout="wide")

# ====== MODEL LOADING ======
@st.cache_resource
def load_finetuned_shufflenet():
    model = models.shufflenet_v2_x1_0(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("best_shufflenet.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_shufflenet():
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.eval()
    return model

@st.cache_resource
def load_cnn():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.eval()
    return model

# ====== IMAGE TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== PREDICT FUNCTION ======
def predict_image(image, model):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
    return pred_class, probs


# ====== UI ======
st.title("🕵️‍♂️ DeepFake Detection Tool")

# --- Initialize uploader key if not present ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- Upload image ---
uploaded_file = st.file_uploader(
    "📤 Upload an Image", 
    type=["jpg", "jpeg", "png"],
    key=st.session_state.uploader_key
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.session_state.image = image

    # Model choice
    model_choice = st.selectbox("Choose Model", ["Fine-Tuned ShuffleNetV2", "ShuffleNetV2", "CNN"])

    if "model" not in st.session_state or st.session_state.model_choice != model_choice:
        st.session_state.model_choice = model_choice
        if model_choice == "Fine-Tuned ShuffleNetV2":
            st.session_state.model = load_finetuned_shufflenet()
        elif model_choice == "ShuffleNetV2":
            st.session_state.model = load_shufflenet()
        elif model_choice == "CNN":
            st.session_state.model = load_cnn()

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        analyze_clicked = st.button("🔍 Analyze")
    with col2:
        accuracy_clicked = st.button("📊 Show Accuracy")

    col3, col4 = st.columns(2)
    with col3:
        cm_clicked = st.button("🧩 Show Confusion Matrix")
    with col4:
        graph_clicked = st.button("📈 Show Probability Graph")

    # --- Handle actions ---
    if analyze_clicked:
        pred_class, probs = predict_image(st.session_state.image, st.session_state.model)
        st.session_state.pred_result = {
            "class": "Real" if pred_class == 1 else "Fake",
            "confidence": probs[pred_class] * 100,
            "probs": probs
        }
        st.success(f"Prediction: {st.session_state.pred_result['class']} ({st.session_state.pred_result['confidence']:.2f}%)")

    if accuracy_clicked:
        st.info("⚠️ Model accuracy display removed as per request.")

    if cm_clicked:
        cm = np.array([[70, 10], [8, 72]])  # example confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], 
                    cbar=False, linewidths=1, linecolor='white')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

    if graph_clicked and "pred_result" in st.session_state:
        probs = st.session_state.pred_result["probs"]
        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], probs, color=["red", "green"])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)

    # --- Reset button (only after prediction) ---
    if "pred_result" in st.session_state:
        if st.button("🔄 Reset & Upload New Image"):
            for key in ["image", "pred_result", "model", "model_choice"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.uploader_key += 1  # reset file uploader
            st.rerun()
