from labels import LETTERS, LETTER_TO_INDEX, INDEX_TO_LETTER


def test_letters_covers_alphabet_a_to_z():
    assert LETTERS == [chr(ord("A") + i) for i in range(26)]


def test_letter_to_index_and_back_round_trip():
    for index, letter in enumerate(LETTERS):
        assert LETTER_TO_INDEX[letter] == index
        assert INDEX_TO_LETTER[index] == letter
