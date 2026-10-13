import argparse
import os
import random
import re
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_label_prefix(path: str) -> int:
    name = os.path.basename(path)
    if "_" not in name:
        raise ValueError(f"Unexpected filename format: {name}")
    return int(name.split("_", 1)[0])


def parse_label_raw(path: str, index: int) -> int:
    name = os.path.basename(path)
    base = os.path.splitext(name)[0]
    parts = re.split(r"[_-]+", base)
    nums = []
    for part in parts:
        if part.isdigit():
            nums.append(int(part))
        else:
            break
    if len(nums) < 2:
        raise ValueError(f"Unexpected raw filename format: {name}")
    return nums[index]


def collect_images(data_dir: str, recursive: bool = False) -> List[str]:
    exts = (".jpg", ".jpeg", ".png")
    paths = []
    if recursive:
        for dirpath, _, filenames in os.walk(data_dir):
            for name in filenames:
                if name.lower().endswith(exts):
                    paths.append(os.path.join(dirpath, name))
    else:
        for name in os.listdir(data_dir):
            if name.lower().endswith(exts):
                paths.append(os.path.join(data_dir, name))
    return sorted(paths)


class CloudDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        label_mode: str,
        label_index: int,
        transform: transforms.Compose,
    ) -> None:
        self.paths = paths
        self.label_mode = label_mode
        self.label_index = label_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.label_mode == "prefix":
            label = parse_label_prefix(path)
        else:
            label = parse_label_raw(path, self.label_index)
        return img, int(label) - 1  # zero-based


def build_dataloaders(
    data_dir: str,
    label_mode: str,
    label_index: int,
    extra_train_dir: str,
    extra_label_mode: str,
    extra_label_index: int,
    img_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    data_dir = os.path.abspath(data_dir)
    extra_train_dir = os.path.abspath(extra_train_dir) if extra_train_dir else ""
    paths = [os.path.abspath(p) for p in collect_images(data_dir, recursive=False)]
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    rng = np.random.default_rng(seed)
    labels = []
    for p in paths:
        if label_mode == "prefix":
            labels.append(parse_label_prefix(p))
        else:
            labels.append(parse_label_raw(p, label_index))
    labels = np.array(labels)

    idx = np.arange(len(paths))
    train_idx = []
    test_idx = []
    for cls in np.unique(labels):
        cls_idx = idx[labels == cls]
        rng.shuffle(cls_idx)
        split = int(0.7 * len(cls_idx))
        train_idx.extend(cls_idx[:split])
        test_idx.extend(cls_idx[split:])

    train_paths = [paths[i] for i in train_idx]
    test_paths = [paths[i] for i in test_idx]

    if extra_train_dir:
        extra_paths = [os.path.abspath(p) for p in collect_images(extra_train_dir, recursive=False)]
        if extra_paths:
            train_paths.extend(extra_paths)

    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if extra_train_dir:
        mixed = []
        for p in train_paths:
            if p.startswith(extra_train_dir):
                mixed.append((p, extra_label_mode, extra_label_index))
            else:
                mixed.append((p, label_mode, label_index))

        class MixedDataset(Dataset):
            def __init__(self, items: List[Tuple[str, str, int]], transform):
                self.items = items
                self.transform = transform

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                path, mode, idx_label = self.items[idx]
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                if mode == "prefix":
                    label = parse_label_prefix(path)
                else:
                    label = parse_label_raw(path, idx_label)
                return img, int(label) - 1

        train_ds = MixedDataset(mixed, train_tf)
    else:
        train_ds = CloudDataset(train_paths, label_mode, label_index, train_tf)

    test_ds = CloudDataset(test_paths, label_mode, label_index, test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    freeze_epochs: int,
    out_path: str,
    label_smoothing: float,
    mixup_alpha: float,
    use_amp: bool,
    class_weights: torch.Tensor,
) -> None:
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_acc = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(images.size(0), device=device)
                mixed = lam * images + (1 - lam) * images[perm]
                targets_a = labels
                targets_b = labels[perm]
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(mixed)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc = correct / max(total, 1)
        print(f"Epoch {epoch}/{epochs} - loss {train_loss:.4f} - train_acc {train_acc:.4f} - test_acc {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), out_path)

        if epoch == freeze_epochs:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - epoch))
        scheduler.step()

    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Saved best model to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN training for cloud classification")
    parser.add_argument(
        "--task",
        choices=["cloud_amount", "cloud_type", "raw_cloud_amount", "raw_cloud_type"],
        default="cloud_type",
    )
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--extra-train-dir", default="")
    parser.add_argument("--extra-label-mode", choices=["prefix", "raw"], default="raw")
    parser.add_argument("--extra-label-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="cnn_cloud_model.pth")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    if args.task in ("raw_cloud_amount", "raw_cloud_type"):
        label_mode = "raw"
        label_index = 1 if args.task == "raw_cloud_amount" else 0
        default_dir = os.path.join("data", "labeled_raw")
    else:
        label_mode = "prefix"
        label_index = 0
        default_dir = os.path.join("data", "cloud_amount") if args.task == "cloud_amount" else os.path.join(
            "data", "cloud_type"
        )

    data_dir = args.data_dir if args.data_dir else default_dir
    set_seed(args.seed)
    args.pretrained = not args.no_pretrained

    train_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        label_mode=label_mode,
        label_index=label_index,
        extra_train_dir=args.extra_train_dir,
        extra_label_mode=args.extra_label_mode,
        extra_label_index=args.extra_label_index,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=6, pretrained=args.pretrained).to(device)
    if args.freeze_epochs > 0:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    # Compute class weights from training set
    class_counts = np.zeros(6, dtype=np.int64)
    for _, labels in train_loader:
        for v in labels.numpy():
            class_counts[v] += 1
    class_counts = np.maximum(class_counts, 1)
    weights = class_counts.sum() / class_counts
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_epochs=args.freeze_epochs,
        out_path=args.output,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        use_amp=args.amp and device.type == "cuda",
        class_weights=weights,
    )


if __name__ == "__main__":
    main()
