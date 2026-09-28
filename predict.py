"""Predict the fruit in a single image.

Example:
    python predict.py --image my_apple.jpg
"""
import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from data import get_transforms
from model import get_device, load_checkpoint


def predict(image_path, checkpoint="models/fruit_classifier.pth", top_k=3):
    """Return a list of (class_name, probability) for the top_k predictions."""
    device = get_device()
    model, class_names, img_size = load_checkpoint(checkpoint, device)

    transform = get_transforms(img_size, train=False)
    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)[0]

    top_probs, top_idx = probs.topk(min(top_k, len(class_names)))
    return [(class_names[int(i)], float(p)) for p, i in zip(top_probs, top_idx)]


def main():
    parser = argparse.ArgumentParser(description="Predict a fruit from an image")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--checkpoint", default="models/fruit_classifier.pth")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    results = predict(args.image, args.checkpoint, args.top_k)
    print(f"\nPredictions for {args.image}:")
    for rank, (name, prob) in enumerate(results, start=1):
        print(f"  {rank}. {name:<30s} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
