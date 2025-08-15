import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

# ======= CONFIG ===========
model_path = "best_shufflenet.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= LOAD MODEL ========
@st.cache_resource
def load_model():
    model = models.shufflenet_v2_x1_0(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ======= IMAGE TRANSFORM =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======= STREAMLIT INTERFACE =======
st.title("🕵️‍♂️ DeepFake Detector")
st.write("Upload an image to check if it's **Real** or **Fake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_names = ['Fake', 'Real']
        pred_class = class_names[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()] * 100

    st.markdown(f"### 🧠 Prediction: **{pred_class}** ({confidence:.2f}%)")
