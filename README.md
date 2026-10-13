# Allsky Cloud Lab

**English** | [简体中文](README.zh-CN.md)

All-sky image classification with PyTorch transfer learning, batch inference and an interactive browser demo.

## Interactive Demo

[Open the live demo](https://genlier.github.io/allsky-cloud-lab/). English is the default; use the language selector at the top right to switch to Chinese. The app remembers your explicit choice and preserves the photo and analysis when switching languages.

Upload a sky photo for an automatic cloud-cover estimate, then click classify to obtain scores for six cloud types. The app also offers mask previews, threshold adjustment and JSON export. To run locally, execute `python -m http.server 8000 --directory docs` in the project root and open `http://localhost:8000`.

Separate ResNet18 models classify six cloud types and four cloud-amount classes. The browser runs both ONNX models locally and provides an adjustable color threshold for exploring candidate cloud fractions. See the [ONNX Runtime Web documentation](https://onnxruntime.ai/docs/get-started/with-javascript/web.html).

First use downloads two ONNX models from this site (about 23 MB total) and a pinned ONNX runtime from jsDelivr. Threshold analysis remains available with limited network access. Models are cached in the browser; inference time depends on your device. JSON labels follow the selected language, with stable one-based class IDs for CNN candidates.

## Evaluation

| Task | PyTorch | Deployed ONNX |
| --- | ---: | ---: |
| Cloud type | 67.0% | 67.0% |
| Cloud amount | 80.2% | 81.3% |

Retained test samples: 97 for cloud type and 96 for cloud amount. [Full metrics and configurations](docs/models/model-card.json).

## Installation

```sh
conda activate myenv
python -m pip install -r requirements.txt
```

Keep your existing CUDA-enabled PyTorch installation if available. The code selects CUDA or CPU automatically.

## Data

Place your datasets under `data/`. The repository includes a synthetic demo and browser-inference model parameters.

| Directory | Purpose | Label Format |
| --- | --- | --- |
| `data/cloud_type/` | Supervised cloud-type samples | `class_id.jpg`, classes 1–6 |
| `data/cloud_amount/` | Supervised cloud-amount samples | `class_id.jpg`, classes 1–4 |
| `data/labeled_raw/` | Optional additional labeled samples | First two filename fields: cloud type, then cloud amount; verify before use |
| `data/AllskyImagerData/` | Images for inference; monthly/daily subdirectories supported | No labels required |

Cloud types, in order: clear sky, cirrus, stratiform, cumuliform, cumulonimbus and mixed clouds. Cloud-amount classes: clear, few clouds, cloudy and overcast. Here, `class_id.jpg` denotes a class prefix followed by an image identifier, for example `1_001.jpg`.

`peridata_*.csv` contains accompanying instrument records. The current CNN uses images only. Field units, device codes and time references need verification before record fusion; time matching alone does not supply ground-truth cloud labels.

The original Allsky download contains nested monthly ZIP archives. Extract a small sample from one month to check the pipeline:

```sh
python prepare_allsky.py --archive /path/to/AllskyImagerData.zip --month 202303 --max-images 32
```

Omit `--month` and `--max-images` to extract all months. The script streams one monthly archive at a time, retains accompanying CSVs for selected dates, and checks for path traversal and symbolic links. Allow enough disk space for full extraction. Unlabeled images support inference checks; accuracy is evaluated on a separate labeled test set.

## Training

```sh
python train_cnn.py --data-dir data/cloud_type --task cloud_type --epochs 20 --output outputs/cloud_type.pth
```

Include additional labeled samples:

```sh
python train_cnn.py --data-dir data/cloud_type --extra-data-dir data/labeled_raw --extra-label-mode raw --task cloud_type --output outputs/cloud_type.pth
```

After combining the main and additional samples, decoded-pixel SHA-256 hashes remove exact duplicates. A stratified split assigns 70% to a training pool and 30% to the test set; a validation set is then held out from the training pool. Only validation performance selects the best epoch, and the final test set is evaluated once. Identical pixels with conflicting labels raise an error for review. Near-duplicates, simultaneous images from different devices and adjacent observations still require date/event-grouped evaluation. The current random split describes performance under this specific protocol.

The cloud-amount dataset includes one group with identical pixels and conflicting labels. The demo model explicitly used `--drop-conflicting` to exclude every copy in this group. Without that flag, conflicting labels still raise an error.

Defaults are ImageNet-pretrained ResNet18, mild augmentation, AdamW, cosine learning-rate decay and label smoothing. Use `--model resnet50` for the larger architecture; larger models do not necessarily improve small-sample accuracy. First training downloads public pretrained weights into `.cache/`. Reduce `--batch-size` if GPU memory is limited. Checkpoints, epoch logs, split manifests and final evaluation JSON are written to `outputs/`.

## Small-Data Tuning

```sh
python tune_cnn.py --task cloud_type --data-dir data/cloud_type --extra-data-dir data/labeled_raw --split-file outputs/cloud_type.split.json --batch-size 8 --output-dir outputs/tuning/cloud_type
python tune_cnn.py --task cloud_amount --data-dir data/cloud_amount --extra-data-dir data/labeled_raw --split-file outputs/cloud_amount.split.json --drop-conflicting --batch-size 8 --output-dir outputs/tuning/cloud_amount
```

Run in the CUDA-enabled `myenv` environment after generating the split manifests. Three-fold stratified cross-validation uses only the original training and validation pool. Four ResNet18 configurations compare the baseline, a frozen backbone, last-block fine-tuning and full fine-tuning with frozen BatchNorm. Configuration selection uses mean validation accuracy, with macro F1 as a tie-break. Median best-epoch counts determine final training lengths before a single held-out test comparison of the selected configuration and a refitted baseline. Both refits use the same enlarged training pool. Logs, selected settings, checkpoints and comparisons are saved under the output directory; existing checkpoints and browser models remain unchanged. Previously inspected test sets provide a retrospective comparison, rather than fresh independent confirmation.

Generate each task's manifest with the same main and supplementary datasets. For cloud amount:

```sh
python train_cnn.py --task cloud_amount --data-dir data/cloud_amount --extra-data-dir data/labeled_raw --drop-conflicting --epochs 12 --output outputs/cloud_amount.pth
```

The deployed models use the selected `full_frozen_bn` configuration: backbone learning rate `3e-5`, head learning rate `3e-4`, weight decay `1e-3`, label smoothing `0.05`, batch size 8 and frozen BatchNorm statistics. The final cloud-type and cloud-amount refits use 10 and 5 epochs, respectively. The [model card](docs/models/model-card.json) records cross-validation results, PyTorch and deployed ONNX test metrics, training counts and file checksums.

## Browser Model Export

```sh
python -m pip install -r requirements-export.txt
python export_onnx.py --weights outputs/tuning/cloud_type/full_frozen_bn.pth --output docs/models/cloud_type.onnx --quantize-weights
python export_onnx.py --weights outputs/tuning/cloud_amount/full_frozen_bn.pth --output docs/models/cloud_amount.onnx --quantize-weights
```

Both browser models use ResNet18 with 224-pixel inputs; retain this configuration when retraining demo models. Export removes docstrings and metadata. FP32 export verifies numerical agreement. The deployed models use symmetric per-output-channel INT8 quantization of convolutional and linear weights, with FP32 activations, reducing file size to about one quarter. Converted-model test results are recorded in `docs/models/model-card.json`. Browser Canvas resizing and training-time Pillow resizing can produce small interpolation differences.

Evaluate a converted model with the fixed split:

```sh
python evaluate_onnx.py --model docs/models/cloud_type.onnx --task cloud_type --data-dir data/cloud_type --extra-data-dir data/labeled_raw --split-file outputs/cloud_type.split.json --output outputs/cloud_type.onnx.metrics.json
```

## Batch Inference

```sh
python predict_cnn.py --weights outputs/cloud_type.pth --data-dir data/AllskyImagerData --output-csv outputs/predictions.csv
```

The script scans recursively and writes relative paths, predicted classes, maximum softmax probabilities and read status to CSV. Damaged images receive an error row while processing continues. Scores are uncalibrated model outputs. Load model files from trusted sources. For legacy plain `state_dict` weights, specify the architecture with `--model resnet18` or `--model resnet50`.

## Mathematical Methods

ResNet maps images to features, and a linear layer produces class logits. Softmax gives

$$p_k = \frac{\exp(z_k)}{\sum_j\exp(z_j)}.$$

Label smoothing uses $q_k=(1-\varepsilon)\mathbf{1}_{k=y}+\varepsilon/K$ and optimizes cross-entropy $L=-\sum_k q_k\log p_k$. Cosine learning-rate decay reduces the update magnitude toward the end of training.

The threshold demo uses $D=(R-B)/(R+B+\epsilon)$. Pixels satisfying $D>t$ in the valid region are cloud candidates; their fraction determines the four amount classes. With the circular option enabled, only the valid circular region is considered. Thin clouds, aerosols, sun glare and lens projection affect this pixel fraction, which differs from calibrated hemispheric cloud coverage.
