import argparse

import torch
from torch.utils.data import DataLoader

from dataset import LandmarkDataset
from model import ASLClassifier
from labels import LETTERS, INDEX_TO_LETTER

def evaluate(model_path, validation_csv):
    model = ASLClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dataset = LandmarkDataset(validation_csv)
    loader = DataLoader(dataset, batch_size=32)

    correct = 0
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    return correct / len(dataset)

def per_letter_accuracy(model_path, validation_csv):
    model = ASLClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dataset = LandmarkDataset(validation_csv)
    loader = DataLoader(dataset, batch_size=32)

    correct = {letter: 0 for letter in LETTERS}
    total = {letter: 0 for letter in LETTERS}

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            predictions = outputs.argmax(dim=1)
            for true_idx, pred_idx in zip(labels.tolist(), predictions.tolist()):
                true_letter = INDEX_TO_LETTER[true_idx]
                total[true_letter] += 1
                if true_idx == pred_idx:
                    correct[true_letter] += 1

    return {
        letter: (correct[letter] / total[letter] if total[letter] else None)
        for letter in LETTERS
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/asl_model.pt")
    parser.add_argument("--validation-csv", default="data/validation_data.csv")
    args = parser.parse_args()

    accuracy = evaluate(args.model, args.validation_csv)
    print(f"real world validation accuracy: {accuracy:.2%}")

    print("\nper letter breakdown:")
    breakdown = per_letter_accuracy(args.model, args.validation_csv)
    for letter, acc in sorted(breakdown.items(), key=lambda item: (item[1] is None, item[1])):
        if acc is None:
            print(f"  {letter}: no samples collected")
        else:
            print(f"  {letter}: {acc:.0%}")