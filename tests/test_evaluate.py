import csv

import torch

from evaluate import evaluate
from model import ASLClassifier


def test_evaluate_returns_accuracy_in_valid_range(tmp_path):
    csv_path = tmp_path / "val.csv"
    header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
    rows = [[0.1] * 63 + ["A"], [0.2] * 63 + ["B"], [0.3] * 63 + ["C"]]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    model = ASLClassifier()
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    accuracy = evaluate(str(model_path), str(csv_path))

    assert 0.0 <= accuracy <= 1.0