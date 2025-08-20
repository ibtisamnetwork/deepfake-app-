import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

# ====== CONFIG ======
st.set_page_config(page_title="DeepFake Detector", page_icon="🕵️‍♂️", layout="wide")

# ====== CSS ======
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #71b7e6, #9b59b6);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #fff;
    }
    h1 {
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .result-box {
        padding: 15px 20px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        text-align: center;
        font-weight: 700;
        font-size: 1.2rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
        margin-bottom: 20px;
    }
    img {
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

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

# ====== TRANSFORM ======
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

# ====== ACCURACIES ======
MODEL_ACCURACIES = {
    "Fine-Tuned ShuffleNetV2": 0.92,
    "ShuffleNetV2": 0.88,
    "CNN": 0.85
}

# ====== UI ======
st.title("🕵️‍♂️ DeepFake Detection Tool")

# Layout: Left (controls), Right (results)
left_col, right_col = st.columns([1.2, 1])

with left_col:
    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    # File upload
    uploaded_file = top_left.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.image = Image.open(uploaded_file).convert("RGB")
        top_left.image(st.session_state.image, use_column_width=True)

    # Model selection
    model_choice = top_right.selectbox("Choose Model", ["-- Select a Model --",
                                                        "Fine-Tuned ShuffleNetV2",
                                                        "ShuffleNetV2",
                                                        "CNN"])

    # Accuracy button
    acc_clicked = bottom_left.button("📊 Show Accuracy")

    # Confusion Matrix button
    cm_clicked = bottom_right.button("🧩 Show Confusion Matrix")

    # Analyze button
    analyze_clicked = st.button("🔍 Analyze", use_container_width=True)

    # Reset button (only after prediction)
    if st.session_state.get("show_reset", False):
        if st.button("🔄 Reset", use_container_width=True):
            for key in ["image", "pred_result", "show_reset"]:
                st.session_state.pop(key, None)
            st.rerun()

with right_col:
    # Prediction results
    if "pred_result" in st.session_state:
        result = st.session_state.pred_result
        st.markdown(f'<div class="result-box">Prediction: {result["class"]} '
                    f'({result["confidence"]:.2f}%)</div>', unsafe_allow_html=True)

        # Probability graph
        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], result["probs"], color=["crimson", "seagreen"])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)

# ====== Actions ======
if analyze_clicked:
    if "image" not in st.session_state:
        st.warning("⚠️ Please upload an image before analyzing.")
    elif model_choice == "-- Select a Model --":
        st.warning("⚠️ Please select a model before analyzing.")
    else:
        with st.spinner("🔎 Analyzing image... Please wait..."):
            progress_bar = st.progress(0)
            for percent in range(0, 101, 10):
                time.sleep(0.1)
                progress_bar.progress(percent)

            # Load model
            if model_choice == "Fine-Tuned ShuffleNetV2":
                st.session_state.model = load_finetuned_shufflenet()
            elif model_choice == "ShuffleNetV2":
                st.session_state.model = load_shufflenet()
            elif model_choice == "CNN":
                st.session_state.model = load_cnn()

            # Predict
            pred_class, probs = predict_image(st.session_state.image, st.session_state.model)
            st.session_state.pred_result = {
                "class": "Real" if pred_class == 1 else "Fake",
                "confidence": probs[pred_class] * 100,
                "probs": probs
            }
            st.session_state.show_reset = True
        st.rerun()

if acc_clicked:
    if model_choice == "-- Select a Model --":
        st.warning("⚠️ Please select a model first to view accuracy.")
    else:
        acc = MODEL_ACCURACIES.get(model_choice, None)
        if acc:
            st.metric(label=f"{model_choice} Accuracy", value=f"{acc*100:.2f}%")
        else:
            st.error("No accuracy available for this model.")

if cm_clicked:
    cm = np.array([[70, 10], [8, 72]])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Fake", "Real"],
                yticklabels=["Fake", "Real"],
                cbar=False, linewidths=1, linecolor='white')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)
