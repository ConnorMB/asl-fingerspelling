import csv
import random

from train_model import train


def _write_synthetic_csv(path, letters, samples_per_letter):
    header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
    rng = random.Random(0)
    rows = []
    for letter in letters:
        for _ in range(samples_per_letter):
            rows.append([rng.random() for _ in range(63)] + [letter])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_train_runs_end_to_end_and_saves_artifacts(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_synthetic_csv(csv_path, ["A", "B"], samples_per_letter=20)

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    accuracy = train(str(csv_path), output_dir=str(output_dir), epochs=1, batch_size=4)

    assert 0.0 <= accuracy <= 1.0
    assert (output_dir / "asl_model.pt").exists()
    assert (output_dir / "loss_curve.png").exists()