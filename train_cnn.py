import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from cloud_core import LABELS, LabeledDataset, images, model, pixel_hash


def collect(root, task, mode):
    items = []
    for path in images(root):
        parts = re.split(r"[_-]+", path.stem)
        position = 1 if mode == "raw" and task == "cloud_amount" else 0
        try:
            label = int(parts[position]) - 1
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid labeled filename; check label format") from exc
        if not 0 <= label < len(LABELS[task]):
            raise ValueError("Label outside the selected task range")
        items.append((path, label))
    return items


def deduplicate(items, drop_conflicting=False):
    unique = {}
    conflicts = set()
    for path, label in items:
        digest = pixel_hash(path)
        if digest in unique and unique[digest][1] != label:
            if not drop_conflicting:
                raise ValueError("Identical image pixels have conflicting labels; audit local data")
            conflicts.add(digest)
        unique.setdefault(digest, (path, label))
    if conflicts:
        print(f"excluded_conflicting_pixel_groups={len(conflicts)}")
    return [item for digest, item in unique.items() if digest not in conflicts]


@torch.inference_mode()
def evaluate(net, loader, device):
    net.eval()
    truth, predicted = [], []
    for x, y in loader:
        predicted.extend(net(x.to(device)).argmax(1).cpu().tolist())
        truth.extend(y.tolist())
    return truth, predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--task", choices=LABELS, default="cloud_type")
    parser.add_argument("--label-mode", choices=["prefix", "raw"], default="prefix")
    parser.add_argument("--extra-data-dir")
    parser.add_argument("--extra-label-mode", choices=["prefix", "raw"], default="raw")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--drop-conflicting", action="store_true", help="Exclude all copies of conflicting pixel groups")
    parser.add_argument("--output", default="outputs/cloud_model.pth")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.img_size) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, img-size and lr must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.hub.set_dir(str(Path(__file__).parent / ".cache" / "torch"))
    items = collect(args.data_dir, args.task, args.label_mode)
    if args.extra_data_dir:
        items += collect(args.extra_data_dir, args.task, args.extra_label_mode)
    original_count = len(items)
    items = deduplicate(items, args.drop_conflicting)
    if not items:
        raise ValueError("No labeled images found")
    try:
        train, test = train_test_split(items, test_size=0.3, random_state=args.seed,
                                      stratify=[label for _, label in items])
        train, valid = train_test_split(train, test_size=0.2, random_state=args.seed,
                                       stratify=[label for _, label in train])
    except ValueError as exc:
        raise ValueError("Insufficient samples for stratified train/validation/test split") from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Content IDs preserve the split without exposing local paths.
    split_rows = [{"split": split, "image_id": pixel_hash(path), "label": label + 1}
                  for split, group in [("train", train), ("validation", valid), ("test", test)]
                  for path, label in group]
    output.with_suffix(".split.json").write_text(json.dumps(split_rows, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = [DataLoader(LabeledDataset(group, args.img_size, augment),
                         batch_size=args.batch_size, shuffle=augment,
                         num_workers=args.num_workers, pin_memory=device.type == "cuda")
               for group, augment in [(train, True), (valid, False), (test, False)]]
    net = model(args.model, len(LABELS[args.task]), not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    print(f"{device} | unique={len(items)} duplicates={original_count-len(items)} | train={len(train)} val={len(valid)} test={len(test)}")
    with output.with_suffix(".history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "loss", "validation_accuracy"])
        for epoch in range(args.epochs):
            net.train()
            loss_total = 0.0
            for x, y in loaders[0]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = criterion(net(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_total += loss.item() * len(y)
            truth, predicted = evaluate(net, loaders[1], device)
            accuracy = accuracy_score(truth, predicted)
            writer.writerow([epoch + 1, loss_total / len(train), accuracy])
            stream.flush()
            print(f"{epoch+1}/{args.epochs} loss={loss_total/len(train):.4f} val={accuracy:.4f}")
            if accuracy > best:
                best = accuracy
                torch.save({"state_dict": net.state_dict(), "model": args.model,
                            "task": args.task, "labels": LABELS[args.task],
                            "img_size": args.img_size}, output)
            scheduler.step()
    checkpoint = torch.load(output, map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["state_dict"])
    truth, predicted = evaluate(net, loaders[2], device)
    indices = list(range(len(LABELS[args.task])))
    metrics = {"test_accuracy": accuracy_score(truth, predicted), "best_validation_accuracy": best,
               "confusion_matrix": confusion_matrix(truth, predicted, labels=indices).tolist(),
               "report": classification_report(truth, predicted, labels=indices,
                   target_names=LABELS[args.task], output_dict=True, zero_division=0)}
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test={metrics['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
