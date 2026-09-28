"""Train a fruit classifier with transfer learning.

Example:
    python train.py --data_dir data/fruits-360 --epochs 5
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from data import build_dataloaders
from model import build_model, get_device, SUPPORTED_MODELS


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """One pass over the loader. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def plot_history(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a fruit recognition model")
    parser.add_argument("--data_dir", required=True,
                        help="Folder that contains the 'Training' and 'Test' folders")
    parser.add_argument("--model", default="resnet18", choices=SUPPORTED_MODELS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=100)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Only train the new classification head (faster, less accurate)")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.val_split, args.num_workers
    )
    print(f"Classes: {len(class_names)} | train batches: {len(train_loader)} "
          f"| val batches: {len(val_loader)} | test batches: {len(test_loader)}")

    model = build_model(args.model, len(class_names), pretrained=True,
                        freeze_backbone=args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model_dir, output_dir = Path(args.model_dir), Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "fruit_classifier.pth"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device,
                                          optimizer, desc=f"Epoch {epoch} [train]")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device,
                                      desc=f"Epoch {epoch} [val]")
        scheduler.step()

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{time.time() - start:.0f}s")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "model_name": args.model,
                "img_size": args.img_size,
            }, ckpt_path)
            print(f"  -> saved best model to {ckpt_path}")

    # Final check on the held-out test set using the best checkpoint
    best = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device, desc="Test")
    print(f"\nTest loss {test_loss:.4f} | Test accuracy {test_acc:.4f}")

    history["test_loss"], history["test_acc"] = test_loss, test_acc
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, output_dir / "training_curves.png")
    print(f"Saved training curves and history to {output_dir}/")


if __name__ == "__main__":
    main()
