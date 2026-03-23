import pandas as pd
import os
import re
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- Config ---
PROJECT_ROOT = Path("/home/ed_st/code/anteecip/computer_vision/plant_detect")
IMG_SIZE = 256        # EfficientNet expects 224×224
BATCH_SIZE = 32
EPOCHS_FROZEN = 10    # train head only
EPOCHS_FINE = 20      # fine-tune top layers

# --- Load labels, extract species from folder name ---
# labels = pd.read_csv(PROJECT_ROOT / "data/labels.csv")
# labels["species"] = labels["filename"].apply(lambda f: Path(f).parts[2])  # data/raw/<species>/...
base_dir = "./backend/data/all_images"

rows = []
for fichier in sorted(os.listdir(base_dir)):
    if not fichier.lower().endswith(".jpg"):
        continue

    # Supprime _x.jpg final pour extraire le label
    label = re.sub(r'_\d+\.jpg$', '', fichier, flags=re.IGNORECASE)

    rows.append({
        "label" : label,
        "path"  : fichier
    })

labels = pd.DataFrame(rows, columns=["label", "path"])


print(labels["label"].value_counts())


# --- Build tf.data pipeline (loads & resizes on-the-fly, no RAM explosion) ---
le = LabelEncoder()
labels["label"] = le.fit_transform(labels["label"])
NUM_CLASSES = len(le.classes_)

train_df, test_df = train_test_split(labels, test_size=0.15, stratify=labels["label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=42)

def make_dataset(df, augment=False):
    def load(row):
        path = str(PROJECT_ROOT / row["path"])
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0  # normalize to [0, 1]
        return img, row["label"]

    paths = df["path"].tolist()
    labs  = df["label"].tolist()
    ds = tf.data.Dataset.from_tensor_slices({"path": paths, "label": labs})
    ds = ds.map(lambda r: load(r), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (
            tf.image.random_flip_left_right(
            tf.image.random_brightness(
            tf.image.random_contrast(x, 0.8, 1.2),
            0.1)), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df, augment=True)
val_ds   = make_dataset(val_df)
test_ds  = make_dataset(test_df)

# --- Phase 1: train classification head, backbone frozen ---
base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN)

# --- Phase 2: unfreeze top 30 layers and fine-tune with low LR ---
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
