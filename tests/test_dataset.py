import csv

import torch

from dataset import LandmarkDataset
from labels import LETTER_TO_INDEX


def _write_csv(path, rows):
    header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_dataset_length_matches_row_count(tmp_path):
    csv_path = tmp_path / "data.csv"
    rows = [[0.1] * 63 + ["A"], [0.2] * 63 + ["B"], [0.3] * 63 + ["A"]]
    _write_csv(csv_path, rows)

    dataset = LandmarkDataset(str(csv_path))

    assert len(dataset) == 3


def test_dataset_returns_features_and_correct_label_index(tmp_path):
    csv_path = tmp_path / "data.csv"
    rows = [[0.1] * 63 + ["A"], [0.2] * 63 + ["C"]]
    _write_csv(csv_path, rows)

    dataset = LandmarkDataset(str(csv_path))
    features, label = dataset[1]

    assert isinstance(features, torch.Tensor)
    assert features.shape == (63,)
    assert label.item() == LETTER_TO_INDEX["C"]