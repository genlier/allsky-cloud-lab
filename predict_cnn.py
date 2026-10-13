import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from cloud_core import LABELS, images, model, transform


class InferenceDataset(Dataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.tf = transform(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as source:
                return self.tf(source.convert("RGB")), str(path), "ok"
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return None, str(path), "unreadable"


def collate(batch):
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--task", choices=LABELS, default="cloud_type")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--recursive", action="store_true", help="Compatibility flag; scan is always recursive")
    args = parser.parse_args()
    root = Path(args.data_dir).resolve()
    paths = images(root)
    if not paths:
        raise ValueError("No images found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        architecture, task, size = checkpoint["model"], checkpoint["task"], checkpoint["img_size"]
        state = checkpoint["state_dict"]
    else:
        architecture, task, size = args.model, args.task, args.img_size
        state = checkpoint
    labels = LABELS[task]
    # Historical cloud-amount checkpoints also used a six-output head.
    classes = state["fc.weight"].shape[0]
    if classes != len(labels):
        raise ValueError("Checkpoint output count differs from task; select the matching task")
    net = model(architecture, classes)
    net.load_state_dict(state)
    net.to(device).eval()
    loader = DataLoader(InferenceDataset(paths, size), batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=collate)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    processed = skipped = 0
    with output.open("w", encoding="utf-8-sig", newline="") as stream, torch.inference_mode():
        writer = csv.writer(stream)
        writer.writerow(["path", "pred_label", "label_name", "prob", "status"])
        for batch in loader:
            good = [item for item in batch if item[0] is not None]
            predictions = iter([])
            if good:
                probs = torch.softmax(net(torch.stack([item[0] for item in good]).to(device)), 1)
                confidence, predicted = probs.max(1)
                predictions = iter(zip(predicted.cpu().tolist(), confidence.cpu().tolist()))
            for image, path, status in batch:
                relative = Path(path).relative_to(root).as_posix()
                if image is None:
                    writer.writerow([relative, "", "", "", status])
                    skipped += 1
                else:
                    label, confidence = next(predictions)
                    writer.writerow([relative, label + 1, labels[label], f"{confidence:.6f}", status])
                processed += 1
            stream.flush()
            if processed % (args.batch_size * 20) == 0:
                print(f"{processed}/{len(paths)} unreadable={skipped}")
    print(f"processed={processed} unreadable={skipped}")


if __name__ == "__main__":
    main()
