import torch

from model import ASLClassifier


def test_forward_pass_output_shape():
    model = ASLClassifier()
    batch = torch.randn(8, 63)

    output = model(batch)

    assert output.shape == (8, 26)


def test_forward_pass_with_custom_sizes():
    model = ASLClassifier(input_size=10, num_classes=4)
    batch = torch.randn(3, 10)

    output = model(batch)

    assert output.shape == (3, 4)
