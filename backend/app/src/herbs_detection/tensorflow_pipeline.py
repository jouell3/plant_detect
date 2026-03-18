from __future__ import annotations

from pathlib import Path

from herbs_detection.load_data import split_image_dataframe
from herbs_detection.params import (
    ALL_IMAGES_DIR,
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    IMG_SIZE,
    MODEL_DIR,
    TENSORFLOW_WEIGHTS_PATH,
)


def _build_tf_dataset(dataframe, image_size: int, batch_size: int, training: bool):
    import tensorflow as tf

    paths = dataframe["image_path"].tolist()
    labels = dataframe["label"].tolist()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [image_size, image_size])
        # EfficientNet in TF 2.15 already contains its own input rescaling layer.
        image = tf.cast(image, tf.float32)
        image.set_shape((image_size, image_size, 3))
        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
            ]
        )
        dataset = dataset.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_datasets(
    image_dir: Path = ALL_IMAGES_DIR,
    image_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    random_state: int = 42,
):
    train_df, val_df, test_df, label_encoder = split_image_dataframe(
        image_dir=image_dir,
        random_state=random_state,
    )

    return (
        _build_tf_dataset(train_df, image_size=image_size, batch_size=batch_size, training=True),
        _build_tf_dataset(val_df, image_size=image_size, batch_size=batch_size, training=False),
        _build_tf_dataset(test_df, image_size=image_size, batch_size=batch_size, training=False),
        list(label_encoder.classes_),
        {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    )


def build_model(num_classes: int, image_size: int = IMG_SIZE):
    import tensorflow as tf

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model, base_model


def train_tensorflow_pipeline(
    image_dir: Path = ALL_IMAGES_DIR,
    epochs_frozen: int = 10,
    epochs_finetune: int = 20,
    image_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
):
    import tensorflow as tf

    train_ds, val_ds, test_ds, class_names, split_sizes = build_datasets(
        image_dir=image_dir,
        image_size=image_size,
        batch_size=batch_size,
    )
    model, base_model = build_model(len(class_names), image_size=image_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_frozen,
        callbacks=callbacks,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune,
        callbacks=callbacks,
    )

    evaluation = model.evaluate(test_ds, verbose=0, return_dict=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(TENSORFLOW_WEIGHTS_PATH)
    CLASS_NAMES_PATH.write_text("\n".join(class_names) + "\n")

    return {
        "model": model,
        "class_names": class_names,
        "num_classes": len(class_names),
        "split_sizes": split_sizes,
        "evaluation": evaluation,
        "history_frozen": history_frozen.history,
        "history_finetune": history_finetune.history,
        "saved_model_path": str(TENSORFLOW_WEIGHTS_PATH),
    }


if __name__ == "__main__":
    results = train_tensorflow_pipeline()
    print(results["evaluation"])
    print(f"Saved TensorFlow model to {results['saved_model_path']}")
