import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cloud_core import LABELS, model, pixel_hash
from train_cnn import collect, deduplicate


@dataclass(frozen=True)
class Configuration:
    name: str
    scope: str
    backbone_lr: float
    head_lr: float
    freeze_bn: bool
    weight_decay: float = 1e-4
    smoothing: float = 0.05
    augmentation: str = "mild"


CONFIGURATIONS = [
    Configuration("baseline", "all", 1e-4, 1e-4, False, augmentation="original"),
    Configuration("head_only", "head", 0, 3e-3, True),
    Configuration("last_block", "last", 3e-5, 3e-4, True),
    Configuration("full_frozen_bn", "all", 3e-5, 3e-4, True, weight_decay=1e-3),
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(items, manifest):
    rows = json.loads(Path(manifest).read_text(encoding="utf-8"))
    entries = {row["image_id"]: row for row in rows}
    if len(entries) != len(rows):
        raise ValueError("Split manifest contains repeated image IDs")
    found = {pixel_hash(path): (path, label) for path, label in items}
    if set(found) != set(entries):
        raise ValueError("Dataset content differs from the fixed split manifest")
    pool, test = [], []
    for row in rows:
        item = found[row["image_id"]]
        if item[1] + 1 != row["label"]:
            raise ValueError("Dataset labels differ from the fixed split manifest")
        if row["split"] in {"train", "validation"}:
            pool.append(item)
        elif row["split"] == "test":
            test.append(item)
        else:
            raise ValueError("Unknown split name")
    if not pool or not test:
        raise ValueError("Training pool and test set must both be nonempty")
    return pool, test


class CachedDataset(Dataset):
    def __init__(self, items, cache, augmentation="none"):
        self.items, self.cache = items, cache
        ops = []
        if augmentation == "original":
            ops = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                   transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1)]
        elif augmentation == "mild":
            ops = [transforms.RandomHorizontalFlip(),
                   transforms.Pad(12, padding_mode="reflect"),
                   transforms.RandomRotation(10), transforms.CenterCrop(224)]
        elif augmentation != "none":
            raise ValueError("Unknown augmentation")
        self.tf = transforms.Compose(ops + [transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        return self.tf(self.cache[path]), label


def parameter_groups(net, config):
    for name, parameter in net.named_parameters():
        parameter.requires_grad = name.startswith("fc.") or config.scope == "all" or (
            config.scope == "last" and name.startswith("layer4."))
    backbone = [p for name, p in net.named_parameters() if p.requires_grad and not name.startswith("fc.")]
    groups = [{"params": net.fc.parameters(), "lr": config.head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": config.backbone_lr})
    return groups


def set_train_mode(net, config):
    net.train()
    # Frozen blocks keep their pretrained BatchNorm running statistics.
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            frozen_block = config.scope == "head" or (config.scope == "last" and not name.startswith("layer4."))
            if config.freeze_bn or frozen_block:
                module.eval()


@torch.inference_mode()
def predict(net, loader, device):
    net.eval()
    truth, probabilities = [], []
    for x, y in loader:
        probabilities.extend(net(x.to(device)).softmax(1).cpu().numpy())
        truth.extend(y.tolist())
    return np.asarray(truth), np.asarray(probabilities)


def metrics(truth, probabilities, classes):
    predicted = probabilities.argmax(1)
    indices = list(range(classes))
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=indices, average="macro", zero_division=0)),
            "loss": float(-np.log(np.maximum(probabilities[np.arange(len(truth)), truth], 1e-8)).mean())}


def fit(config, train, valid, cache, classes, device, epochs, batch_size, seed, history):
    seed_everything(seed)
    net = model("resnet18", classes, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(net, config), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.smoothing)
    train_loader = DataLoader(CachedDataset(train, cache, config.augmentation), batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(CachedDataset(valid, cache), batch_size=batch_size) if valid else None
    best_key, best_metrics, best_epoch = None, None, epochs
    with history.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "validation_accuracy", "validation_macro_f1", "validation_loss"])
        for epoch in range(1, epochs + 1):
            set_train_mode(net, config)
            loss_total = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = criterion(net(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_total += loss.item() * len(y)
            score = metrics(*predict(net, valid_loader, device), classes) if valid_loader else None
            writer.writerow([epoch, loss_total / len(train)] +
                            ([score["accuracy"], score["macro_f1"], score["loss"]] if score else ["", "", ""]))
            stream.flush()
            if score:
                key = (score["accuracy"], score["macro_f1"], -score["loss"])
                if best_key is None or key > best_key:
                    best_key, best_metrics, best_epoch = key, score, epoch
            scheduler.step()
    # CV only records validation results; final refits never select epochs on the test set.
    return net, best_metrics, best_epoch


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--extra-data-dir")
    parser.add_argument("--task", choices=LABELS, required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--drop-conflicting", action="store_true")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.folds < 2 or min(args.epochs, args.batch_size) < 1:
        parser.error("folds >= 2 and positive epochs/batch-size required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.hub.set_dir(str(Path(__file__).parent / ".cache" / "torch"))
    device = torch.device("cuda")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "experiment.json", {"arguments": vars(args),
        "configurations": [asdict(config) for config in CONFIGURATIONS],
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0)})
    items = collect(args.data_dir, args.task, "prefix")
    if args.extra_data_dir:
        items += collect(args.extra_data_dir, args.task, "raw")
    items = deduplicate(items, args.drop_conflicting)
    pool, test = load_split(items, args.split_file)
    classes = len(LABELS[args.task])
    if min(np.bincount([label for _, label in pool], minlength=classes)) < args.folds:
        raise ValueError("Insufficient samples per class for cross-validation")
    cache = {}
    resize = transforms.Resize((224, 224))
    for path, _ in pool + test:
        with Image.open(path) as image:
            cache[path] = resize(image.convert("RGB"))
    splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(
        np.zeros(len(pool)), [label for _, label in pool]))
    print(f"{args.task} | {torch.cuda.get_device_name(0)} | pool={len(pool)} test={len(test)} folds={args.folds}", flush=True)
    summaries = []
    for config in CONFIGURATIONS:
        folds = []
        for fold, (train_ids, valid_ids) in enumerate(splits):
            net, score, epoch = fit(config, [pool[i] for i in train_ids], [pool[i] for i in valid_ids],
                cache, classes, device, args.epochs, args.batch_size, args.seed + fold,
                output / f"{config.name}.fold{fold + 1}.csv")
            del net
            torch.cuda.empty_cache()
            folds.append({"fold": fold + 1, "best_epoch": epoch, **score})
            write_json(output / f"{config.name}.cv.json", {"configuration": asdict(config), "folds": folds})
            print(f"{config.name} fold={fold+1} acc={score['accuracy']:.4f} f1={score['macro_f1']:.4f} epoch={epoch}", flush=True)
        summaries.append({"configuration": asdict(config), "folds": folds,
            "mean_accuracy": float(np.mean([row["accuracy"] for row in folds])),
            "std_accuracy": float(np.std([row["accuracy"] for row in folds])),
            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in folds])),
            "refit_epochs": int(np.median([row["best_epoch"] for row in folds]))})
        write_json(output / "cv_results.json", summaries)
    winner = max(summaries, key=lambda row: (row["mean_accuracy"], row["mean_macro_f1"], -row["std_accuracy"]))
    write_json(output / "selection.json", {"selected": winner, "selection": "mean CV accuracy, macro F1 tie-break; test untouched"})
    print(f"selected={winner['configuration']['name']} cv={winner['mean_accuracy']:.4f}", flush=True)
    # Lock both configurations and epoch counts before the single final test comparison.
    final_models = {}
    for row in summaries:
        if row["configuration"]["name"] not in {"baseline", winner["configuration"]["name"]}:
            continue
        config = Configuration(**row["configuration"])
        net, _, _ = fit(config, pool, None, cache, classes, device, row["refit_epochs"],
            args.batch_size, args.seed, output / f"{config.name}.refit.csv")
        checkpoint_path = output / f"{config.name}.pth"
        torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "model": "resnet18",
                    "task": args.task, "labels": LABELS[args.task], "img_size": 224}, checkpoint_path)
        final_models[config.name] = net.cpu()
        del net
    test_loader = DataLoader(CachedDataset(test, cache), batch_size=args.batch_size)
    final = {}
    for name, net in final_models.items():
        truth, probabilities = predict(net.to(device), test_loader, device)
        score = metrics(truth, probabilities, classes)
        score["confusion_matrix"] = confusion_matrix(truth, probabilities.argmax(1), labels=range(classes)).tolist()
        score["report"] = classification_report(truth, probabilities.argmax(1), labels=range(classes),
            target_names=LABELS[args.task], output_dict=True, zero_division=0)
        final[name] = score
        net.cpu()
        print(f"final {name} test={score['accuracy']:.4f} macro_f1={score['macro_f1']:.4f}", flush=True)
    write_json(output / "final_comparison.json", {"task": args.task, "selected": winner["configuration"]["name"],
        "pool_size": len(pool), "test_size": len(test), "results": final,
        "protocol": "existing test held out of CV; refit epoch count fixed by median CV best epoch; test used once for final comparison",
        "limitation": "test results were previously inspected in the project; near duplicates are not event-grouped"})


if __name__ == "__main__":
    main()
