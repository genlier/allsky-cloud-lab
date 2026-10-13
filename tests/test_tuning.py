import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from cloud_core import model
from tune_cnn import CONFIGURATIONS, CachedDataset, load_split, parameter_groups, set_train_mode


class TuningTests(unittest.TestCase):
    def test_trainable_blocks_and_batchnorm(self):
        for config in CONFIGURATIONS:
            net = model("resnet18", 6)
            parameter_groups(net, config)
            set_train_mode(net, config)
            for name, parameter in net.named_parameters():
                expected = name.startswith("fc.") or config.scope == "all" or (
                    config.scope == "last" and name.startswith("layer4."))
                self.assertEqual(parameter.requires_grad, expected)
            self.assertTrue(net.fc.training)
            if config.freeze_bn:
                self.assertTrue(all(not module.training for module in net.modules()
                    if isinstance(module, torch.nn.BatchNorm2d)))

    def test_split_keeps_test_out_and_checks_labels(self):
        rows = [{"image_id": "a", "label": 1, "split": "train"},
                {"image_id": "b", "label": 2, "split": "validation"},
                {"image_id": "c", "label": 1, "split": "test"}]
        items = [("a", 0), ("b", 1), ("c", 0)]
        with tempfile.TemporaryDirectory() as directory, patch("tune_cnn.pixel_hash", side_effect=lambda path: path):
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            pool, test = load_split(items, path)
            self.assertEqual(pool, items[:2])
            self.assertEqual(test, items[2:])
            with self.assertRaises(ValueError):
                load_split(items + [("d", 0)], path)
            with self.assertRaises(ValueError):
                load_split([("a", 1), *items[1:]], path)

    def test_cached_image_remains_unchanged(self):
        image = Image.new("RGB", (224, 224), (50, 130, 220))
        before = image.tobytes()
        for augmentation in ["none", "original", "mild"]:
            dataset = CachedDataset([("sample", 0)], {"sample": image}, augmentation)
            tensor, label = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 224, 224))
            self.assertEqual(label, 0)
            self.assertEqual(image.tobytes(), before)


if __name__ == "__main__":
    unittest.main()
