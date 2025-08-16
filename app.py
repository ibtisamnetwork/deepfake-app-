import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import time
import pandas as pd
from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# ====== CUSTOM CSS ====== (keep your existing CSS here, omitted for brevity)

st.markdown("""
<style>
    /* ... your existing CSS styles here ... */
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
st.markdown("""
<div class="header">
    <h1>🕵️‍♂️ DeepFake Detector</h1>
    <p>Upload an image and let AI detect if it's <strong>Real</strong> or <strong>Fake</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ====== FILE UPLOAD ======
uploaded_file = st.file_uploader("📤 Choose an image file", type=["jpg", "jpeg", "png"])

# Tagline below uploader (always visible)
st.markdown(
    '<p class="tagline">Upload a face image to detect deepfakes — stay aware!</p>',
    unsafe_allow_html=True
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='🖼 Uploaded Image')

    analyze = st.button("Analyze")

    if analyze:
        with st.spinner("Analyzing picture..."):
            progress_bar = st.progress(0)
            img_tensor = transform(image).unsqueeze(0).to("cpu")

            for percent in range(0, 101, 20):
                time.sleep(0.15)
                progress_bar.progress(percent)

            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                _, predicted = torch.max(outputs, 1)
                class_names = ['Fake', 'Real']
                pred_class = class_names[predicted.item()]
                confidence = probabilities[predicted.item()] * 100

            progress_bar.progress(100)
            time.sleep(0.2)

            # Show probability distribution bar chart
            df = pd.DataFrame({'Class': class_names, 'Probability': probabilities}).set_index('Class')
            st.bar_chart(df)

            # Grad-CAM for explainability
            # IMPORTANT: Find the correct last conv layer name for ShuffleNetV2
            cam_extractor = GradCAM(model, target_layer="conv5")  # <-- You might need to adjust this!

            activation_map = cam_extractor(predicted.item(), outputs)
            heatmap = activation_map[0].cpu()

            # Convert heatmap to PIL image resized to original image
            heatmap_img = to_pil_image(heatmap).resize(image.size, resample=Image.BILINEAR).convert("RGBA")

            # Overlay heatmap on original image with transparency
            overlay = Image.blend(image.convert("RGBA"), heatmap_img, alpha=0.5)

            st.image(overlay, caption="Grad-CAM Heatmap Overlay")

            # Display prediction result box
            color_class = "pred-real" if pred_class == "Real" else "pred-fake"
            st.markdown(
                f"""
                <div class="result-box">
                    <span>🧠 Prediction:</span> <span class="{color_class}">{pred_class}</span>
                    <p>Confidence: <strong>{confidence:.2f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Footer disclaimer
st.markdown("<div class='footer'>🔍 This result is based on the uploaded image and may not be perfect. Always verify with additional tools.</div>", unsafe_allow_html=True)
