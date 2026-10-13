import hashlib
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

LABELS = {
    "cloud_type": ["晴空", "卷云", "层状云", "积状云", "积雨云", "混合云"],
    "cloud_amount": ["无云", "少云", "多云", "阴天"],
}
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def images(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("Image directory does not exist")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def transform(size, augment=False):
    ops = [transforms.Resize((size, size))]
    if augment:
        ops += [transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)]
    return transforms.Compose(ops + [transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def model(name, classes, pretrained=False):
    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("Unsupported architecture")
    net.fc = torch.nn.Linear(net.fc.in_features, classes)
    return net


def pixel_hash(path):
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


class LabeledDataset(Dataset):
    def __init__(self, items, size, augment=False):
        self.items = items
        self.tf = transform(size, augment)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as source:
            image = self.tf(source.convert("RGB"))
        return image, label
