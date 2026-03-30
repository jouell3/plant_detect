import csv
import io
from pathlib import Path

import streamlit as st

from i18n import get_language, render_language_selector
# Local imports for validation
from utils import validate_images_batch, show_validation_errors, show_validation_summary

st.set_page_config(page_title="Label Images", layout="wide")

GRID_COLS = 5
GRID_ROWS = 10
PAGE_SIZE = GRID_COLS * GRID_ROWS  # 50

# ---------------------------------------------------------------------------
# CSS: green buttons for "good" images, muted for unlabeled
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    div[data-testid="stButton"] button[kind="secondary"] {
        width: 100%;
        background-color: #444;
        color: #ccc;
        border: 1px solid #666;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        border: 1px solid #1b5e20;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_labels_from_upload(uploaded_file) -> dict[str, str]:
    content = uploaded_file.read().decode("utf-8")
    return {row["filename"]: row["label"] for row in csv.DictReader(io.StringIO(content))}


def labels_to_csv(labels: dict[str, str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["filename", "label", "name"])
    writer.writeheader()
    for filename, label in sorted(labels.items()):
        name = filename.split("_")[0]
        writer.writerow({"filename": filename, "label": label, "name": name})
    return output.getvalue()


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "label_image_files" not in st.session_state:
    st.session_state.label_image_files = []  # list of {"name": str, "bytes": bytes}
if "label_page" not in st.session_state:
    st.session_state.label_page = 0
if "labels" not in st.session_state:
    st.session_state.labels = {}
if "label_uploader_key" not in st.session_state:
    st.session_state.label_uploader_key = 0

with st.sidebar:
    render_language_selector()

lang = get_language()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Progress")
    if st.session_state.label_image_files:
        total = len(st.session_state.label_image_files)
        good = sum(1 for v in st.session_state.labels.values() if v == "good")
        labeled = len(st.session_state.labels)
        st.metric("Total images" if lang == "en" else "Total images", total)
        st.metric("Labeled" if lang == "en" else "Labelees", labeled)
        st.metric("Good" if lang == "en" else "Bonnes", good)
        if st.session_state.labels:
            st.download_button(
                "⬇ Download labels CSV" if lang == "en" else "⬇ Telecharger le CSV des labels",
                data=labels_to_csv(st.session_state.labels),
                file_name="labels.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("Load images to start." if lang == "en" else "Chargez des images pour commencer.")

# ---------------------------------------------------------------------------
# Header + file inputs
# ---------------------------------------------------------------------------
st.title("Image Labeling" if lang == "en" else "Labellisation d'images")

st.markdown(
    "Browse images from your dataset folders and select the ones that will be used to train your model. "
    "Labels can be exported to a reusable CSV file for training or retraining."
    if lang == "en"
    else "Vous pouvez parcourir les images de votre dataset par dossier, et selectionner les images qui serviront a l'entrainement de votre modele. "
    "Les labels pourront etre exportes dans un fichier CSV reutilisable pour entrainer ou re-entrainer votre modele."
)

st.markdown(
    "To begin, upload your images (.jpg or .jpeg) using the button below. "
    "You can also upload an existing labels CSV to pre-fill selections (optional)."
    if lang == "en"
    else "Pour commencer, uploadez vos images (format .jpg ou .jpeg) via le bouton ci-dessous. "
    "Vous pouvez egalement uploader un fichier CSV de labels deja existant pour pre-remplir les selections (optionnel)."
)

col_path, col_btn = st.columns([5, 1])
with col_path:
    uploaded_images = st.file_uploader(
        label="Upload images (.jpg / .jpeg)" if lang == "en" else "Uploader des images (.jpg / .jpeg)",
        label_visibility="collapsed",
        accept_multiple_files=True,
        type=["jpg", "jpeg"],
        key=f"img_uploader_{st.session_state.label_uploader_key}",
    )
    uploaded_labels = st.file_uploader(
        label="Labels CSV (optional)" if lang == "en" else "CSV de labels (optionnel)",
        label_visibility="collapsed",
        type=["csv"],
        key=f"csv_uploader_{st.session_state.label_uploader_key}",
    )
with col_btn:
    load_clicked = st.button("Load", use_container_width=True)

if load_clicked:
    if not uploaded_images:
        st.error("Please upload at least one image." if lang == "en" else "Veuillez uploader au moins une image.")
    else:
        # Validate images before processing
        valid_files, invalid_files = validate_images_batch(uploaded_images)
        
        # Show validation results
        show_validation_summary(len(valid_files), len(uploaded_images))
        show_validation_errors(invalid_files)
        
        if valid_files:
            # Read bytes into session state so they survive reruns
            st.session_state.label_image_files = [
                {"name": f.name, "bytes": f.read()} for f in valid_files
            ]
            st.session_state.label_page = 0
            st.session_state.label_uploader_key += 1  # reset uploaders on next render
            if uploaded_labels is not None:
                st.session_state.labels = load_labels_from_upload(uploaded_labels)
            else:
                st.session_state.labels = {}
        else:
            st.session_state.label_image_files = []

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
if not st.session_state.label_image_files:
    st.stop()

total_pages = max(1, (len(st.session_state.label_image_files) + PAGE_SIZE - 1) // PAGE_SIZE)
page = st.session_state.label_page
start = page * PAGE_SIZE
page_files = st.session_state.label_image_files[start : start + PAGE_SIZE]

for row in range(GRID_ROWS):
    cols = st.columns(GRID_COLS)
    for col_idx in range(GRID_COLS):
        img_idx = row * GRID_COLS + col_idx
        if img_idx >= len(page_files):
            break
        file = page_files[img_idx]
        key = file["name"]
        label = st.session_state.labels.get(key, "not_selected")

        with cols[col_idx]:
            st.image(file["bytes"], width="stretch", caption=file["name"])

            is_good = label == "good"
            btn_label = "✅ Good" if is_good else ("○ Keep?" if lang == "en" else "○ Garder ?")
            btn_type = "primary" if is_good else "secondary"

            if st.button(
                btn_label,
                key=f"btn_{start + img_idx}",
                type=btn_type,
                use_container_width=True,
            ):
                st.session_state.labels[key] = "not_selected" if is_good else "good"
                st.rerun()

# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
st.divider()
p_left, p_mid, p_right = st.columns([1, 2, 1])

with p_left:
    if st.button("← Prev", disabled=(page == 0), use_container_width=True):
        st.session_state.label_page -= 1
        st.rerun()

with p_mid:
    end_img = min(start + PAGE_SIZE, len(st.session_state.label_image_files))
    st.metric(
        label="Progress" if lang == "en" else "Progression",
        value=f"Page {page + 1} / {total_pages}",
        delta=f"images {start + 1}-{end_img}",
    )
    target_page = st.number_input(
        "Go to page" if lang == "en" else "Aller a la page",
        min_value=1,
        max_value=total_pages,
        value=page + 1,
        step=1,
        key="label_jump_page",
    )
    if target_page != page + 1:
        st.session_state.label_page = int(target_page) - 1
        st.rerun()

with p_right:
    if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True):
        st.session_state.label_page += 1
        st.rerun()
