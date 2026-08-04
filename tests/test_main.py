import main
from labels import LETTERS
from model import ASLClassifier


def test_predict_letter_returns_a_valid_letter():
    main.model = ASLClassifier()
    main.model.eval()

    vector = [0.0] * 63
    letter = main.predict_letter(vector)

    assert letter in LETTERS