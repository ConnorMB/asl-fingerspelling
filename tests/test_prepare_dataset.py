import csv

import cv2
import numpy as np

from prepare_dataset import process_dataset


def _make_fake_images(input_dir, letter, count):
    letter_dir = input_dir / letter
    letter_dir.mkdir(parents=True)
    for i in range(count):
        path = letter_dir / f"{letter}{i}.jpg"
        cv2.imwrite(str(path), np.zeros((10, 10, 3), dtype=np.uint8))


def test_process_dataset_respects_samples_per_letter_limit(tmp_path):
    input_dir = tmp_path / "input"
    _make_fake_images(input_dir, "A", 5)
    _make_fake_images(input_dir, "B", 5)
    output_csv = tmp_path / "out.csv"

    def fake_extractor(image):
        return [0.1] * 63

    counts = process_dataset(
        str(input_dir), str(output_csv), samples_per_letter=2, extract_landmarks=fake_extractor
    )

    assert counts["A"] == 2
    assert counts["B"] == 2
    assert counts["C"] == 0


def test_process_dataset_writes_expected_row_count(tmp_path):
    input_dir = tmp_path / "input"
    _make_fake_images(input_dir, "A", 3)
    output_csv = tmp_path / "out.csv"

    def fake_extractor(image):
        return [0.5] * 63

    process_dataset(
        str(input_dir), str(output_csv), samples_per_letter=3, extract_landmarks=fake_extractor
    )

    with open(output_csv) as f:
        rows = list(csv.reader(f))

    assert len(rows) == 1 + 3  # header + 3 data rows
    assert rows[0][-1] == "label"
    assert rows[1][-1] == "A"


def test_process_dataset_skips_images_where_extractor_finds_no_hand(tmp_path):
    input_dir = tmp_path / "input"
    _make_fake_images(input_dir, "A", 3)
    output_csv = tmp_path / "out.csv"

    def fake_extractor(image):
        return None

    counts = process_dataset(
        str(input_dir), str(output_csv), samples_per_letter=3, extract_landmarks=fake_extractor
    )

    assert counts["A"] == 0