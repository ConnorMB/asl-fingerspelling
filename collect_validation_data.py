import csv
import os

import cv2
import mediapipe as mp

from labels import LETTERS
from landmarks import landmarks_to_vector

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_FILE = "data/validation_data.csv"


def init_data_file():
    if not os.path.exists(DATA_FILE):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        header.append("label")
        with open(DATA_FILE, "w", newline="") as f:
            csv.writer(f).writerow(header)


def main():
    init_data_file()

    hands = mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    counts = {letter: 0 for letter in LETTERS}

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, "press a-z = record letter  q = quit", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"samples this session: {sum(counts.values())}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Collect ASL Validation Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif chr(key).isalpha() and chr(key).upper() in LETTERS and results.multi_hand_landmarks:
            letter = chr(key).upper()
            vector = landmarks_to_vector(results.multi_hand_landmarks[0])
            with open(DATA_FILE, "a", newline="") as f:
                csv.writer(f).writerow(vector + [letter])
            counts[letter] += 1

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()