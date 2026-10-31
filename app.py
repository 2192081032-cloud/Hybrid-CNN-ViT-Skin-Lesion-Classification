import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import HybridCNNViT

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬"
)

st.title("Skin Cancer Detection Using Hybrid CNN-ViT")

st.write(
    "Welcome to our AI-powered skin lesion "
    "classification application."
)

# Seven HAM10000 lesion classes
class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# Load trained model
@st.cache_resource
def load_model():
    model = HybridCNNViT(num_classes=7)

    model_path = (
        Path(__file__).resolve().parent
        / "hybrid_cnn_vit_best.pth"
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Upload image
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Skin Lesion Image"
    )

    if st.button("Predict Skin Lesion"):

        try:
            model = load_model()

            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)

            with torch.inference_mode():
                outputs = model(image_tensor)

                probabilities = F.softmax(
                    outputs,
                    dim=1
                )

                confidence, predicted = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_class = class_names[
                predicted.item()
            ]

            confidence_score = (
                confidence.item() * 100
            )

            st.success(
                f"Predicted Lesion: {predicted_class}"
            )

            st.write(
                f"Model Confidence: "
                f"{confidence_score:.2f}%"
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.warning(
    "This application is a research prototype. "
    "It is not a medical diagnostic tool. "
    "Predictions should not be used to diagnose "
    "skin cancer or make treatment decisions."
)