"""Evaluate a trained model on the test set.

Example:
    python evaluate.py --data_dir data/fruits-360
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from data import build_test_loader
from model import get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fruit classifier")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", default="models/fruit_classifier.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    device = get_device()
    model, class_names, img_size = load_checkpoint(args.checkpoint, device)
    loader, test_classes = build_test_loader(args.data_dir, img_size,
                                             args.batch_size, args.num_workers,
                                             class_names=class_names)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            preds = model(images.to(device)).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    accuracy = (y_pred == y_true).mean()
    print(f"\nTest accuracy: {accuracy:.4f} ({len(y_true)} images)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_ids = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=label_ids,
                                   target_names=class_names, digits=4, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report)
    print(f"Saved per-class report to {output_dir / 'classification_report.txt'}")

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    # Most common mistakes
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    order = np.argsort(off_diag.ravel())[::-1][:10]
    print("\nMost frequent confusions (true -> predicted : count)")
    for idx in order:
        i, j = np.unravel_index(idx, off_diag.shape)
        if off_diag[i, j] == 0:
            break
        print(f"  {class_names[i]} -> {class_names[j]} : {off_diag[i, j]}")

    # Confusion matrix image (row-normalised; class names omitted because there are too many)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues")
    ax.set_title("Confusion matrix (row-normalised)")
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
