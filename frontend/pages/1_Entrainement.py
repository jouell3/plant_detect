import csv
import io

import streamlit as st

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

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Progress")
    if st.session_state.label_image_files:
        total = len(st.session_state.label_image_files)
        good = sum(1 for v in st.session_state.labels.values() if v == "good")
        labeled = len(st.session_state.labels)
        st.metric("Total images", total)
        st.metric("Labeled", labeled)
        st.metric("Good", good)
        if st.session_state.labels:
            st.download_button(
                "⬇ Download labels CSV",
                data=labels_to_csv(st.session_state.labels),
                file_name="labels.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("Load images to start.")

# ---------------------------------------------------------------------------
# Header + file inputs
# ---------------------------------------------------------------------------
st.title("Image Labeling")

col_path, col_btn = st.columns([5, 1])
with col_path:
    uploaded_images = st.file_uploader(
        label="Upload images (.jpg / .jpeg)",
        label_visibility="collapsed",
        accept_multiple_files=True,
        type=["jpg", "jpeg"],
    )
    uploaded_labels = st.file_uploader(
        label="Label CSV (optional)",
        label_visibility="collapsed",
        type=["csv"],
    )
with col_btn:
    load_clicked = st.button("Load", use_container_width=True)

if load_clicked:
    if not uploaded_images:
        st.error("Please upload at least one image.")
    else:
        # Read bytes into session state so they survive reruns
        st.session_state.label_image_files = [
            {"name": f.name, "bytes": f.read()} for f in uploaded_images
        ]
        st.session_state.label_page = 0
        if uploaded_labels is not None:
            st.session_state.labels = load_labels_from_upload(uploaded_labels)
            st.success(f"Loaded {len(st.session_state.label_image_files)} images and labels from {uploaded_labels.name}.")
        else:
            st.session_state.labels = {}
            st.success(f"Loaded {len(st.session_state.label_image_files)} images.")

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
            st.image(file["bytes"], width=250, caption=file["name"])

            is_good = label == "good"
            btn_label = "✅ Good" if is_good else "○ Keep?"
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
    st.markdown(
        f"<p style='text-align:center; padding-top:8px;'>"
        f"Page {page + 1} / {total_pages} &nbsp;·&nbsp; images {start + 1}–{end_img}"
        f"</p>",
        unsafe_allow_html=True,
    )

with p_right:
    if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True):
        st.session_state.label_page += 1
        st.rerun()
