from pathlib import Path

import streamlit as st
from herbs_detection.model import predict_top3, load_model
from herbs_detection.model_sklearn import predict_top3 as predict_top3_sklearn

load_model()

st.set_page_config(page_title="Batch Predict", layout="wide")

GRID_COLS = 5
GRID_ROWS = 20
PAGE_SIZE = GRID_COLS * GRID_ROWS  # 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scan_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


@st.cache_data(show_spinner=False)
def cached_predict_top3(img_path: str) -> list[tuple[str, float]]:
    return predict_top3(img_path)


@st.cache_data(show_spinner=False)
def cached_predict_top3_sklearn(img_path: str) -> list[tuple[str, float]]:
    return predict_top3_sklearn(img_path)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []
if "page" not in st.session_state:
    st.session_state.page = 0

# ---------------------------------------------------------------------------
# Header + folder input
# ---------------------------------------------------------------------------
st.title("Batch Predictions")

col_path, col_btn = st.columns([5, 1])
with col_path:
    folder_input = st.text_input(
        "Image folder",
        value="data/raw/all_images",
        label_visibility="collapsed",
        placeholder="Path to folder with .jpg images",
    )
with col_btn:
    load_clicked = st.button("Load", use_container_width=True)

if load_clicked:
    folder = Path(folder_input).expanduser().resolve()
    if not folder.is_dir():
        st.error(f"Folder not found: {folder}")
    else:
        paths = scan_images(folder)
        if not paths:
            st.warning("No image files found in that folder.")
        else:
            st.session_state.image_paths = paths
            st.session_state.page = 0
            st.success(f"Loaded {len(paths)} images.")

if not st.session_state.image_paths:
    st.stop()

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
total_pages = max(1, (len(st.session_state.image_paths) + PAGE_SIZE - 1) // PAGE_SIZE)
page = st.session_state.page
start = page * PAGE_SIZE
page_paths = st.session_state.image_paths[start : start + PAGE_SIZE]

for row in range(GRID_ROWS):
    cols = st.columns(GRID_COLS)
    for col_idx in range(GRID_COLS):
        img_idx = row * GRID_COLS + col_idx
        if img_idx >= len(page_paths):
            break
        path = page_paths[img_idx]

        with cols[col_idx]:
            st.image(str(path), width=250)
            st.caption(path.name)

            with st.spinner(""):
                preds = cached_predict_top3(str(path))
                preds_sklearn = cached_predict_top3_sklearn(str(path))
            st.markdown("Pytorch model predictions:", unsafe_allow_html=True)
            for rank, (species, conf) in enumerate(preds, 1):
                bar_color = "#2e7d32" if rank == 1 else "#555"
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between;
                                font-size:0.78rem; margin-bottom:2px;">
                        <span>{rank}. {species}</span>
                        <span style="color:{bar_color}; font-weight:bold;">{conf:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("Sklearn model predictions:", unsafe_allow_html=True)
            for rank, (species, conf) in enumerate(preds_sklearn, 1):
                bar_color = "#2e7d32" if rank == 1 else "#555"
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between;
                                font-size:0.78rem; margin-bottom:2px;">
                        <span>{rank}. {species}</span>
                        <span style="color:{bar_color}; font-weight:bold;">{conf:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
st.divider()
p_left, p_mid, p_right = st.columns([1, 2, 1])

with p_left:
    if st.button("← Prev", disabled=(page == 0), use_container_width=True):
        st.session_state.page -= 1
        st.rerun()

with p_mid:
    end_img = min(start + PAGE_SIZE, len(st.session_state.image_paths))
    st.markdown(
        f"<p style='text-align:center; padding-top:8px;'>"
        f"Page {page + 1} / {total_pages} &nbsp;·&nbsp; images {start + 1}–{end_img}"
        f"</p>",
        unsafe_allow_html=True,
    )

with p_right:
    if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True):
        st.session_state.page += 1
        st.rerun()
