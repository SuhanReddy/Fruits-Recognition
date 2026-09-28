"""Model definitions (transfer learning) and checkpoint helpers."""
import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = ("resnet18", "mobilenet_v2")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name: str, num_classes: int, pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    """Build an ImageNet-pretrained CNN with a new classification head."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    elif name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        head = model.classifier
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {SUPPORTED_MODELS}.")

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return model


def load_checkpoint(path, device):
    """Load a checkpoint saved by train.py. Returns (model, class_names, img_size)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    class_names = ckpt["class_names"]
    model = build_model(ckpt["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, class_names, ckpt["img_size"]
