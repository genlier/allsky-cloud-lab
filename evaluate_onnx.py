import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cloud_core import LABELS, images, pixel_hash, transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', choices=LABELS, required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--extra-data-dir')
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    paths = images(args.data_dir)
    if args.extra_data_dir:
        paths += images(args.extra_data_dir)
    lookup = {pixel_hash(path): path for path in paths}
    split = json.loads(Path(args.split_file).read_text(encoding='utf-8'))
    selected = [row for row in split if row['split'] == 'test']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(args.model, options, providers=['CPUExecutionProvider'])
    size = session.get_inputs()[0].shape[-1]
    tf = transform(size)
    truth, predicted = [], []
    for row in selected:
        with Image.open(lookup[row['image_id']]) as source:
            tensor = tf(source.convert('RGB')).unsqueeze(0).numpy()
        predicted.append(int(np.argmax(session.run(None, {'image': tensor})[0])))
        truth.append(row['label'] - 1)
    classes = list(range(len(LABELS[args.task])))
    metrics = {'test_accuracy': accuracy_score(truth, predicted),
               'confusion_matrix': confusion_matrix(truth, predicted, labels=classes).tolist(),
               'report': classification_report(truth, predicted, labels=classes,
                    target_names=LABELS[args.task], output_dict=True, zero_division=0)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"test={metrics['test_accuracy']:.4f}")


if __name__ == '__main__':
    main()
