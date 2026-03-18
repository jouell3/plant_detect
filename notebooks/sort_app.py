import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import zipfile

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    model.load_state_dict(torch.load("leaf_focus_resnet18.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def predict_leaf_focus(img: Image.Image):
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    return prob

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌿 Leaf Image Sorter")
st.write("Upload images and automatically sort them into leaf‑focused vs non‑leaf.")

uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

threshold = st.slider("Leaf‑focus threshold", 0.0, 1.0, 0.7, 0.05)

if uploaded_files:
    leaf_images = []
    non_leaf_images = []

    st.write(f"Processing {len(uploaded_files)} images...")

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        prob = predict_leaf_focus(img)

        if prob >= threshold:
            leaf_images.append((file.name, img))
        else:
            non_leaf_images.append((file.name, img))

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("🌱 Leaf‑focused images")
    cols = st.columns(4)
    for i, (name, img) in enumerate(leaf_images):
        cols[i % 4].image(img, caption=f"{name}")

    st.subheader("🌾 Non‑leaf images")
    cols = st.columns(4)
    for i, (name, img) in enumerate(non_leaf_images):
        cols[i % 4].image(img, caption=f"{name}")

    # -----------------------------
    # Download ZIPs
    # -----------------------------
    def make_zip(image_list):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for name, img in image_list:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="JPEG")
                z.writestr(name, img_bytes.getvalue())
        return buf.getvalue()

    st.download_button(
        "Download leaf‑focused images (ZIP)",
        data=make_zip(leaf_images),
        file_name="leaf_images.zip",
        mime="application/zip"
    )

    st.download_button(
        "Download non‑leaf images (ZIP)",
        data=make_zip(non_leaf_images),
        file_name="non_leaf_images.zip",
        mime="application/zip"
    )