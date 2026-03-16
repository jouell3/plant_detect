import csv
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Label Images", layout="wide")

GRID_COLS = 5
GRID_ROWS = 10

PAGE_SIZE = GRID_COLS * GRID_ROWS  # 25

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

def load_labels(folder: Path) -> dict[str, str]:
    csv_path = folder
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        return {Path(row["filename"]).name: row["label"] for row in csv.DictReader(f)}


def save_labels(folder: Path, labels: dict[str, str]) -> None:
    csv_path = folder
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "name"])
        writer.writeheader()
        for filename, label in sorted(labels.items()):
            parts = st.session_state.folder.parts
            data_idx = next((i for i, p in enumerate(parts) if p == "data"), None)
            rel_folder = Path(*parts[data_idx:]) if data_idx is not None else st.session_state.folder
            name = filename.split("_")[0]
            name = name.split("/")[-1]
            writer.writerow({"filename": str(rel_folder / filename), "label": label, "name": name})


def scan_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}
    )



# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []
if "page" not in st.session_state:
    st.session_state.page = 0
if "labels" not in st.session_state:
    st.session_state.labels = {}
if "folder" not in st.session_state:
    st.session_state.folder = None
if "label_file" not in st.session_state:
    st.session_state.label_file = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Progress")
    if st.session_state.image_paths:
        total = len(st.session_state.image_paths)
        good = sum(1 for v in st.session_state.labels.values() if v == "good")
        labeled = len(st.session_state.labels)
        st.metric("Total images", total)
        st.metric("Labeled", labeled)
        st.metric("Good", good)
        st.metric("Not selected", labeled - good)
        if total:
            st.progress(labeled / total, text=f"{labeled}/{total} labeled")
    else:
        st.info("Load a folder to start.")

# ---------------------------------------------------------------------------
# Header + folder input
# ---------------------------------------------------------------------------
st.title("Image Labeling")

col_path, col_btn = st.columns([5, 1])
with col_path:
    folder_input = st.text_input(
        "Image folder",
        value="data/raw/",
        label_visibility="collapsed",
        placeholder="Path to folder with .jpg images",
    )
    label_file = st.text_input(
        "Label file (optional)",
        value="data/labels.csv",
        label_visibility="collapsed",
        placeholder="Path to CSV with filename,label (overrides defaults)",
    )
with col_btn:
    load_clicked = st.button("Load", use_container_width=True)

if load_clicked:
    folder = Path(folder_input).expanduser().resolve()
    label_file = Path(label_file).expanduser().resolve()
    if label_file.is_file():
        st.session_state.label_file = label_file
        st.session_state.labels = {}
        st.session_state.labels = load_labels(label_file)
        st.success(f"Loaded labels from {label_file}")
    if not folder.is_dir():
        st.error(f"Folder not found: {folder}")
    else:
        paths = scan_images(folder)
        if not paths:
            st.warning("No .jpg/.jpeg files found in that folder.")
        else:
            st.session_state.image_paths = paths
            st.session_state.folder = folder
            st.session_state.page = 0
            
            st.success(f"Loaded {len(paths)} images.")

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
if not st.session_state.image_paths:
    st.stop()

total_pages = max(1, (len(st.session_state.image_paths) + PAGE_SIZE - 1) // PAGE_SIZE)
page = st.session_state.page
start = page * PAGE_SIZE
page_paths = st.session_state.image_paths[start : start + PAGE_SIZE]

# Render grid
for row in range(GRID_ROWS):
    cols = st.columns(GRID_COLS)
    for col_idx in range(GRID_COLS):
        img_idx = row * GRID_COLS + col_idx
        if img_idx >= len(page_paths):
            break
        path = page_paths[img_idx]
        label = st.session_state.labels.get(path.name, "not_selected")

        with cols[col_idx]:
            st.image(str(path), width=250, caption=path.name)

            is_good = label == "good"
            btn_label = "✅ Good" if is_good else "○ Keep?"
            btn_type = "primary" if is_good else "secondary"

            if st.button(
                btn_label,
                key=f"btn_{start + img_idx}",
                type=btn_type,
                use_container_width=True,
            ):
                new_label = "not_selected" if is_good else "good"
                st.session_state.labels[path.name] = new_label
                save_labels(st.session_state.label_file, st.session_state.labels)
                st.rerun()

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
