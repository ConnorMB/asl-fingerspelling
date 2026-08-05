import argparse
import csv
import os
import random

import cv2
import mediapipe as mp

from labels import LETTERS
from landmarks import landmarks_to_vector

mp_hands = mp.solutions.hands


def _mediapipe_extractor():
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )

    def extractor(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        return landmarks_to_vector(result.multi_hand_landmarks[0])

    return extractor


def process_dataset(input_dir, output_csv, samples_per_letter=300, seed=42, extract_landmarks=None):
    if extract_landmarks is None:
        extract_landmarks = _mediapipe_extractor()

    random.seed(seed)
    counts = {letter: 0 for letter in LETTERS}

    header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    header.append("label")

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for letter in LETTERS:
            letter_dir = os.path.join(input_dir, letter)
            if not os.path.isdir(letter_dir):
                continue

            filenames = os.listdir(letter_dir)
            random.shuffle(filenames)

            for filename in filenames:
                if counts[letter] >= samples_per_letter:
                    break

                path = os.path.join(letter_dir, filename)
                image = cv2.imread(path)
                if image is None:
                    continue

                vector = extract_landmarks(image)
                if vector is None:
                    continue

                writer.writerow(vector + [letter])
                counts[letter] += 1

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="asl_alphabet_train/asl_alphabet_train")
    parser.add_argument("--output-csv", default="data/train_data.csv")
    parser.add_argument("--samples-per-letter", type=int, default=300)
    args = parser.parse_args()

    counts = process_dataset(args.input_dir, args.output_csv, args.samples_per_letter)
    total = sum(counts.values())
    print(f"Wrote {total} samples to {args.output_csv}")
    for letter, count in counts.items():
        if count < args.samples_per_letter:
            print(f"  warning: {letter} only got {count}/{args.samples_per_letter} samples")