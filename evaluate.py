import argparse

import torch
from torch.utils.data import DataLoader

from dataset import LandmarkDataset
from model import ASLClassifier


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/asl_model.pt")
    parser.add_argument("--validation-csv", default="data/validation_data.csv")
    args = parser.parse_args()

    accuracy = evaluate(args.model, args.validation_csv)
    print(f"Real-world validation accuracy: {accuracy:.2%}")