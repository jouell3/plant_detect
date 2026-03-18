from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from herbs_detection.params import ALL_IMAGES_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FILENAME_PATTERN = re.compile(r"^(?P<species>.+)_(?P<index>\d+)$")


def parse_species_from_path(path: Path) -> str | None:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    match = FILENAME_PATTERN.match(path.stem)
    if not match:
        return None

    return match.group("species")


def list_image_paths(image_dir: Path = ALL_IMAGES_DIR) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and parse_species_from_path(path) is not None
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"No valid herb images found in {image_dir}. "
            "Expected files like thyme_49.jpg or rosemary_97.jpg."
        )

    return image_paths


def build_image_dataframe(image_dir: Path = ALL_IMAGES_DIR) -> pd.DataFrame:
    records = [
        {"image_path": str(path), "species": parse_species_from_path(path)}
        for path in list_image_paths(image_dir)
    ]
    return pd.DataFrame(records)


def fit_label_encoder(records: pd.DataFrame) -> LabelEncoder:
    label_encoder = LabelEncoder()
    label_encoder.fit(records["species"])
    return label_encoder


def split_image_dataframe(
    image_dir: Path = ALL_IMAGES_DIR,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    records = build_image_dataframe(image_dir)
    label_encoder = fit_label_encoder(records)
    encoded_records = records.copy()
    encoded_records["label"] = label_encoder.transform(encoded_records["species"])

    train_df, test_df = train_test_split(
        encoded_records,
        test_size=test_size,
        stratify=encoded_records["species"],
        random_state=random_state,
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["species"],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        label_encoder,
    )
