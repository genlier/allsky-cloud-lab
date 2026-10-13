import argparse
import bisect
import csv
import datetime as dt
import glob
import os
import re
from typing import List, Optional, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _parse_label_prefix(path: str) -> int:
    name = os.path.basename(path)
    if "_" not in name:
        raise ValueError(f"Unexpected filename format: {name}")
    return int(name.split("_", 1)[0])


def _parse_label_raw(path: str, index: int) -> int:
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


PERI_KEYS = ["TMPA", "TMPP", "TMPB", "TMPC", "RHUA", "DWPA"]


def _load_image(path: str, size: int = 128) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _extract_features(
    path: str, peridata: Optional[List[float]] = None, feature_set: str = "basic"
) -> np.ndarray:
    rgb = _load_image(path, size=128)
    rgb_mean = rgb.mean(axis=(0, 1))
    rgb_std = rgb.std(axis=(0, 1))

    hsv = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).convert("HSV"), dtype=np.float32) / 255.0
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    y = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    y_mean = np.array([y.mean()])
    y_std = np.array([y.std()])

    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / 2.0
    gy[1:-1, :] = (y[2:, :] - y[:-2, :]) / 2.0
    grad = np.hypot(gx, gy)
    grad_mean = np.array([grad.mean()])
    grad_std = np.array([grad.std()])
    edge_density = np.array([(grad > 0.1).mean()])

    rgb_sum = rgb.sum(axis=2) + 1e-6
    blue_ratio_mean = np.array([(rgb[:, :, 2] / rgb_sum).mean()])
    red_ratio_mean = np.array([(rgb[:, :, 0] / rgb_sum).mean()])
    sky_index = np.array([((rgb[:, :, 2] - rgb[:, :, 0]) / (rgb[:, :, 2] + rgb[:, :, 0] + 1e-6)).mean()])
    sat_mean = np.array([hsv[:, :, 1].mean()])
    bright_frac = np.array([(y > 0.8).mean()])
    dark_frac = np.array([(y < 0.2).mean()])

    # Center vs edge statistics (capture cloud distribution)
    h, w = y.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = np.sqrt(cy**2 + cx**2)
    center_mask = rr <= (0.35 * r_max)
    ring_mask = (rr > (0.35 * r_max)) & (rr <= (0.7 * r_max))

    def _masked_stats(arr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vals = arr[mask]
        return np.array([vals.mean()], dtype=np.float32), np.array([vals.std()], dtype=np.float32)

    c_r_mean, c_r_std = _masked_stats(rgb[:, :, 0], center_mask)
    c_g_mean, c_g_std = _masked_stats(rgb[:, :, 1], center_mask)
    c_b_mean, c_b_std = _masked_stats(rgb[:, :, 2], center_mask)
    r_r_mean, r_r_std = _masked_stats(rgb[:, :, 0], ring_mask)
    r_g_mean, r_g_std = _masked_stats(rgb[:, :, 1], ring_mask)
    r_b_mean, r_b_std = _masked_stats(rgb[:, :, 2], ring_mask)
    c_y_mean, c_y_std = _masked_stats(y, center_mask)
    r_y_mean, r_y_std = _masked_stats(y, ring_mask)

    base_legacy = np.concatenate(
        [
            rgb_mean,
            rgb_std,
            hsv_mean,
            hsv_std,
            y_mean,
            y_std,
            grad_mean,
            grad_std,
            edge_density,
        ]
    )
    if feature_set in ("legacy", "lbp"):
        base = base_legacy
    else:
        base = np.concatenate(
            [
                base_legacy,
                c_r_mean,
                c_r_std,
                c_g_mean,
                c_g_std,
                c_b_mean,
                c_b_std,
                r_r_mean,
                r_r_std,
                r_g_mean,
                r_g_std,
                r_b_mean,
                r_b_std,
                c_y_mean,
                c_y_std,
                r_y_mean,
                r_y_std,
                blue_ratio_mean,
                red_ratio_mean,
                sky_index,
                sat_mean,
                bright_frac,
                dark_frac,
            ]
        )
    if feature_set == "full":
        h = hsv[:, :, 0].ravel()
        s = hsv[:, :, 1].ravel()
        v = hsv[:, :, 2].ravel()
        h_hist, _ = np.histogram(h, bins=16, range=(0.0, 1.0), density=True)
        s_hist, _ = np.histogram(s, bins=16, range=(0.0, 1.0), density=True)
        v_hist, _ = np.histogram(v, bins=16, range=(0.0, 1.0), density=True)

        gray_small = Image.open(path).convert("L").resize((32, 32))
        gray_small = np.asarray(gray_small, dtype=np.float32).ravel() / 255.0

        base = np.concatenate([base, h_hist, s_hist, v_hist, gray_small])

    if feature_set == "lbp":
        gray = y
        gray_small = Image.open(path).convert("L").resize((64, 64))
        gray_small = np.asarray(gray_small, dtype=np.float32) / 255.0
        c = gray_small[1:-1, 1:-1]
        lbp = (
            ((gray_small[:-2, :-2] >= c) << 7)
            | ((gray_small[:-2, 1:-1] >= c) << 6)
            | ((gray_small[:-2, 2:] >= c) << 5)
            | ((gray_small[1:-1, 2:] >= c) << 4)
            | ((gray_small[2:, 2:] >= c) << 3)
            | ((gray_small[2:, 1:-1] >= c) << 2)
            | ((gray_small[2:, :-2] >= c) << 1)
            | ((gray_small[1:-1, :-2] >= c) << 0)
        )
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
        base = np.concatenate([base, hist.astype(np.float32)])
    if peridata is None:
        return base
    peridata = list(peridata)
    mask = [0.0 if (v is None or np.isnan(v)) else 1.0 for v in peridata]
    filled = [-1.0 if (v is None or np.isnan(v)) else float(v) for v in peridata]
    return np.concatenate([base, np.array(filled, dtype=np.float32), np.array(mask, dtype=np.float32)])


def _parse_timestamp_from_name(name: str) -> Optional[dt.datetime]:
    m = re.search(r"(20\d{12})", name)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")


def _parse_peridata_line(line: str) -> Optional[Tuple[dt.datetime, List[Optional[float]]]]:
    parts = [p.strip() for p in line.split(";") if p.strip()]
    kv = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()
    if "DATE" not in kv or "TIME" not in kv:
        return None
    date_raw = kv["DATE"]  # yy-mm-dd
    time_raw = kv["TIME"]  # HH:MM:SS
    try:
        yy, mm, dd = [int(x) for x in date_raw.split("-")]
        year = 2000 + yy
        hh, mi, ss = [int(x) for x in time_raw.split(":")]
        ts = dt.datetime(year, mm, dd, hh, mi, ss)
    except Exception:
        return None
    values = []
    for key in PERI_KEYS:
        val = kv.get(key)
        if val is None:
            values.append(None)
            continue
        try:
            values.append(float(val))
        except Exception:
            values.append(None)
    return ts, values


def _load_peridata(peridata_path: str) -> Tuple[List[dt.datetime], List[List[Optional[float]]]]:
    times: List[dt.datetime] = []
    values: List[List[Optional[float]]] = []
    with open(peridata_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_peridata_line(line)
            if parsed is None:
                continue
            ts, vals = parsed
            times.append(ts)
            values.append(vals)
    return times, values


def _find_peridata_file(image_path: str, allsky_root: str, timestamp: Optional[dt.datetime]) -> Optional[str]:
    parent = os.path.dirname(image_path)
    local = glob.glob(os.path.join(parent, "peridata_*.csv"))
    if local:
        return sorted(local)[0]
    if timestamp is None:
        return None
    yyyymm = timestamp.strftime("%Y%m")
    yyyymmdd = timestamp.strftime("%Y%m%d")
    day_dir = os.path.join(allsky_root, yyyymm, yyyymmdd)
    candidates = glob.glob(os.path.join(day_dir, "peridata_*.csv"))
    return sorted(candidates)[0] if candidates else None


def _peridata_for_image(
    image_path: str,
    allsky_root: str,
    tolerance_seconds: int,
    cache: dict,
) -> Optional[List[float]]:
    timestamp = _parse_timestamp_from_name(os.path.basename(image_path))
    peridata_path = _find_peridata_file(image_path, allsky_root, timestamp)
    if peridata_path is None:
        return None
    if peridata_path not in cache:
        cache[peridata_path] = _load_peridata(peridata_path)
    times, values = cache[peridata_path]
    if not times or timestamp is None:
        return None
    idx = bisect.bisect_left(times, timestamp)
    candidates = []
    if idx < len(times):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best = None
    best_dt = None
    for i in candidates:
        delta = abs((times[i] - timestamp).total_seconds())
        if best_dt is None or delta < best_dt:
            best_dt = delta
            best = i
    if best_dt is None or best_dt > tolerance_seconds:
        return None
    return values[best]


def _build_dataset(
    data_dir: str,
    label_mode: str,
    label_index: int,
    use_peridata: bool,
    allsky_root: str,
    tolerance_seconds: int,
    feature_set: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    patterns = [os.path.join(data_dir, "*.jpg"), os.path.join(data_dir, "*.png")]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No .jpg files found in {data_dir}")

    features = []
    labels = []
    cache = {}
    for path in paths:
        peridata = None
        if use_peridata:
            peridata = _peridata_for_image(path, allsky_root, tolerance_seconds, cache)
        features.append(_extract_features(path, peridata, feature_set))
        if label_mode == "prefix":
            labels.append(_parse_label_prefix(path))
        else:
            labels.append(_parse_label_raw(path, label_index))

    return np.vstack(features), np.array(labels, dtype=np.int64), paths


def _train_eval(
    data_dir: str,
    test_ratio: float,
    seed: int,
    model_out: str,
    label_mode: str,
    label_index: int,
    use_peridata: bool,
    allsky_root: str,
    tolerance_seconds: int,
    model_type: str,
    feature_set: str,
    svm_c: float,
    svm_gamma: str,
    tune_svm: bool,
    dt_max_depth: int,
    dt_min_samples_leaf: int,
    tune_dt: bool,
    extra_train_dir: str,
    extra_label_mode: str,
    extra_label_index: int,
) -> None:
    x, y, _ = _build_dataset(
        data_dir,
        label_mode,
        label_index,
        use_peridata,
        allsky_root,
        tolerance_seconds,
        feature_set,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    if extra_train_dir:
        x_extra, y_extra, _ = _build_dataset(
            extra_train_dir,
            extra_label_mode,
            extra_label_index,
            use_peridata,
            allsky_root,
            tolerance_seconds,
            feature_set,
        )
        x_train = np.vstack([x_train, x_extra])
        y_train = np.concatenate([y_train, y_extra])

    if model_type == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
            max_features="sqrt",
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400, random_state=seed, n_jobs=1, class_weight="balanced"
        )
    elif model_type == "hgb":
        clf = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=300, random_state=seed
        )
    elif model_type == "decision_tree":
        base = DecisionTreeClassifier(
            random_state=seed,
            class_weight="balanced",
            max_depth=None if dt_max_depth <= 0 else dt_max_depth,
            min_samples_leaf=dt_min_samples_leaf,
        )
        if tune_dt:
            param_grid = {
                "max_depth": [4, 6, 8, 10, 12, None],
                "min_samples_leaf": [1, 2, 4, 8],
                "criterion": ["gini", "entropy", "log_loss"],
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            clf = GridSearchCV(base, param_grid, cv=cv, n_jobs=1, scoring="accuracy")
        else:
            clf = base
    else:
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(C=svm_c, gamma=svm_gamma, class_weight="balanced")),
            ]
        )
        if tune_svm:
            param_grid = {
                "svm__C": [1, 5, 10, 20, 50, 100],
                "svm__gamma": ["scale", "auto", 0.01, 0.05, 0.1],
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            clf = GridSearchCV(base, param_grid, cv=cv, n_jobs=1, scoring="accuracy")
        else:
            clf = base
    clf.fit(x_train, y_train)
    if tune_svm and hasattr(clf, "best_params_"):
        print(f"Best params: {clf.best_params_}")
    if tune_dt and hasattr(clf, "best_params_"):
        print(f"Best params: {clf.best_params_}")

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    if model_out:
        joblib.dump(clf, model_out)
        print(f"Saved model to: {model_out}")


def _predict(
    model_path: str,
    image_path: str,
    use_peridata: bool,
    allsky_root: str,
    tolerance_seconds: int,
    feature_set: str,
) -> None:
    clf = joblib.load(model_path)
    cache = {}
    peridata = None
    if use_peridata:
        peridata = _peridata_for_image(image_path, allsky_root, tolerance_seconds, cache)
    feats = _extract_features(image_path, peridata, feature_set).reshape(1, -1)
    pred = clf.predict(feats)[0]
    proba = clf.predict_proba(feats)[0] if hasattr(clf, "predict_proba") else None
    print(f"Predicted label: {pred}")
    if proba is not None:
        pairs = sorted(zip(clf.classes_, proba), key=lambda x: -x[1])
        print("Top probabilities:")
        for cls, p in pairs:
            print(f"  {cls}: {p:.4f}")


def _iter_images(root: str, recursive: bool) -> List[str]:
    exts = (".jpg", ".jpeg", ".png")
    paths = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(exts):
                    paths.append(os.path.join(dirpath, name))
    else:
        for name in os.listdir(root):
            if name.lower().endswith(exts):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def _predict_dir(
    model_path: str,
    image_dir: str,
    recursive: bool,
    output_csv: str,
    use_peridata: bool,
    allsky_root: str,
    tolerance_seconds: int,
    feature_set: str,
) -> None:
    clf = joblib.load(model_path)
    paths = _iter_images(image_dir, recursive)
    if not paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    rows = []
    cache = {}
    for path in paths:
        peridata = None
        if use_peridata:
            peridata = _peridata_for_image(path, allsky_root, tolerance_seconds, cache)
        feats = _extract_features(path, peridata, feature_set).reshape(1, -1)
        pred = int(clf.predict(feats)[0])
        rows.append((path, pred))

    print(f"Predicted {len(rows)} images.")

    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "pred_label"])
            writer.writerows(rows)
        print(f"Saved predictions to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud amount/type classification")
    parser.add_argument(
        "--task",
        choices=["cloud_amount", "cloud_type", "raw_cloud_amount", "raw_cloud_type"],
        default="cloud_amount",
        help="Choose dataset type for default data directory",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Override dataset directory. Use when data is not in default location.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="cloud_model.joblib")
    parser.add_argument("--predict", default="", help="Predict a single image with a trained model")
    parser.add_argument("--predict-dir", default="", help="Predict all images in a directory")
    parser.add_argument("--recursive", action="store_true", help="Scan predict-dir recursively")
    parser.add_argument("--output-csv", default="", help="Save batch predictions to CSV")
    parser.add_argument("--use-peridata", action="store_true", help="Use peridata_*.csv as extra features")
    parser.add_argument("--allsky-root", default="AllskyImagerData", help="Root for peridata lookup")
    parser.add_argument("--peridata-tolerance-seconds", type=int, default=600)
    parser.add_argument(
        "--model-type",
        choices=["decision_tree", "random_forest", "extra_trees", "svm_rbf", "hgb"],
        default="hgb",
    )
    parser.add_argument(
        "--feature-set", choices=["legacy", "basic", "full", "lbp"], default="legacy"
    )
    parser.add_argument("--svm-c", type=float, default=50.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--tune-svm", action="store_true", help="Grid search for SVM hyperparameters")
    parser.add_argument("--dt-max-depth", type=int, default=8, help="<=0 means unlimited depth")
    parser.add_argument("--dt-min-samples-leaf", type=int, default=2)
    parser.add_argument("--tune-dt", action="store_true", help="Grid search for decision tree hyperparameters")
    parser.add_argument("--extra-train-dir", default="", help="Add extra labeled data to training only")
    parser.add_argument("--extra-label-mode", choices=["prefix", "raw"], default="raw")
    parser.add_argument("--extra-label-index", type=int, default=0)

    args = parser.parse_args()

    if args.predict:
        _predict(
            args.model_out,
            args.predict,
            args.use_peridata,
            args.allsky_root,
            args.peridata_tolerance_seconds,
            args.feature_set,
        )
        return
    if args.predict_dir:
        _predict_dir(
            args.model_out,
            args.predict_dir,
            args.recursive,
            args.output_csv,
            args.use_peridata,
            args.allsky_root,
            args.peridata_tolerance_seconds,
            args.feature_set,
        )
        return

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

    _train_eval(
        data_dir,
        args.test_ratio,
        args.seed,
        args.model_out,
        label_mode,
        label_index,
        args.use_peridata,
        args.allsky_root,
        args.peridata_tolerance_seconds,
        args.model_type,
        args.feature_set,
        args.svm_c,
        args.svm_gamma,
        args.tune_svm,
        args.dt_max_depth,
        args.dt_min_samples_leaf,
        args.tune_dt,
        args.extra_train_dir,
        args.extra_label_mode,
        args.extra_label_index,
    )


if __name__ == "__main__":
    main()
