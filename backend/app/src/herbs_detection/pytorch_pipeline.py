from __future__ import annotations

import pickle
from pathlib import Path

from PIL import Image

from herbs_detection.load_data import split_image_dataframe
from herbs_detection.params import (
    ALL_IMAGES_DIR,
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    IMG_SIZE,
    LABEL_ENCODER_PATH,
    MODEL_DIR,
    PYTORCH_WEIGHTS_PATH,
)


class HerbsDataset:
    def __init__(self, dataframe, transform=None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def build_dataloaders(
    image_dir: Path = ALL_IMAGES_DIR,
    image_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    random_state: int = 42,
):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    train_df, val_df, test_df, label_encoder = split_image_dataframe(
        image_dir=image_dir,
        random_state=random_state,
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        HerbsDataset(train_df, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        HerbsDataset(val_df, transform=eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        HerbsDataset(test_df, transform=eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        label_encoder,
        {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    )


def build_model(num_classes: int):
    from torchvision import models
    import torch

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def _run_epoch(model, dataloader, criterion, device, optimizer=None):
    import torch

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * targets.size(0)
        total_correct += (predictions == targets).sum().item()
        total_examples += targets.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def evaluate_model(model, dataloader, criterion, device):
    return _run_epoch(model, dataloader, criterion, device, optimizer=None)


def train_pytorch_pipeline(
    image_dir: Path = ALL_IMAGES_DIR,
    epochs_head: int = 5,
    epochs_finetune: int = 10,
    image_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    learning_rate_head: float = 1e-3,
    learning_rate_finetune: float = 1e-4,
):
    import torch

    train_loader, val_loader, test_loader, label_encoder, split_sizes = build_dataloaders(
        image_dir=image_dir,
        image_size=image_size,
        batch_size=batch_size,
    )
    class_names = list(label_encoder.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names)).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    history_head = []
    history_finetune = []
    best_state = None
    best_val_accuracy = -1.0

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate_head)
    for epoch in range(epochs_head):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        history_head.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    for parameter in model.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_finetune)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    for epoch in range(epochs_finetune):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])
        history_finetune.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, criterion, device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), PYTORCH_WEIGHTS_PATH)
    with open(LABEL_ENCODER_PATH, "wb") as file:
        pickle.dump(label_encoder, file)
    CLASS_NAMES_PATH.write_text("\n".join(class_names) + "\n")

    return {
        "model": model,
        "class_names": class_names,
        "num_classes": len(class_names),
        "split_sizes": split_sizes,
        "device": str(device),
        "best_val_accuracy": best_val_accuracy,
        "test_metrics": test_metrics,
        "history_head": history_head,
        "history_finetune": history_finetune,
        "saved_model_path": str(PYTORCH_WEIGHTS_PATH),
        "saved_label_encoder_path": str(LABEL_ENCODER_PATH),
    }


if __name__ == "__main__":
    results = train_pytorch_pipeline()
    print(results["test_metrics"])
    print(f"Saved PyTorch model to {results['saved_model_path']}")
