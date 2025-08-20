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

# ====== CUSTOM CSS ======
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #71b7e6, #9b59b6);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #fff;
        padding-bottom: 40px;
    }
    h1 {
        text-align: center;
        font-weight: 700;
        font-size: 2.8rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .uploaded-image {
        display: flex;
        justify-content: center;
        margin-bottom: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        max-width: 450px;
        margin-left: auto;
        margin-right: auto;
    }
    .result-box {
        max-width: 460px;
        margin: 20px auto;
        padding: 20px 25px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        color: #fff;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
    }
    img {
        border-radius: 20px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)


# ====== MODEL LOADING FUNCTIONS ======
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


# ====== UI STARTS ======
st.title("🕵️‍♂️ DeepFake Detection Tool")

# Layout grid
top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)
right_panel = st.container()

# Step 1: Upload Image
with top_left:
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
        st.session_state.image = image

# Step 2: Select Model
with top_right:
    model_choice = st.selectbox("🧠 Choose Model",
                                ["-- Select a Model --", "Fine-Tuned ShuffleNetV2", "ShuffleNetV2", "CNN"])

# Step 3: Action Buttons
with bottom_left:
    analyze_clicked = st.button("🔍 Analyze")
    acc_clicked = st.button("📈 Show Accuracy")

with bottom_right:
    cm_clicked = st.button("🧩 Show Confusion Matrix")

# Reset button appears only after prediction
if st.session_state.get("show_reset", False):
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# Handle button clicks
if analyze_clicked:
    if "image" not in st.session_state:
        st.warning("⚠️ Please upload an image before analyzing.")
    elif model_choice == "-- Select a Model --":
        st.warning("⚠️ Please select a model before analyzing.")
    else:
        with st.spinner("🔎 Analyzing image... Please wait..."):
            progress_bar = st.progress(0)
            for percent in range(0, 101, 10):
                time.sleep(0.1)  # fake delay
                progress_bar.progress(percent)

            # Load selected model
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

        st.success("✅ Analysis complete!")
        st.rerun()

# Right panel results
with right_panel:
    if "pred_result" in st.session_state:
        result = st.session_state.pred_result
        st.markdown(
            f'<div class="result-box">Prediction: {result["class"]} ({result["confidence"]:.2f}%)</div>',
            unsafe_allow_html=True
        )

        # Probability bar graph
        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], result["probs"], color=["red", "green"])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)

# Show Accuracy
if acc_clicked:
    st.info("📊 Accuracy feature will be displayed here (custom per dataset).")

# Show Confusion Matrix
if cm_clicked:
    cm = np.array([[70, 10], [8, 72]])  # Example
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"],
                cbar=False, linewidths=1, linecolor='white')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)
