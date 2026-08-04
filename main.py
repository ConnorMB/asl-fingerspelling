import cv2
import mediapipe as mp
import torch

from labels import INDEX_TO_LETTER
from landmarks import landmarks_to_vector
from model import ASLClassifier

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

model = ASLClassifier()


def load_model(path="data/asl_model.pt"):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()


def predict_letter(vector):
    with torch.no_grad():
        tensor = torch.tensor([vector], dtype=torch.float32)
        outputs = model(tensor)
        index = outputs.argmax(dim=1).item()
    return INDEX_TO_LETTER[index]


def run():
    load_model()
    hands = mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_letter = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            vector = landmarks_to_vector(hand_landmarks)
            current_letter = predict_letter(vector)

        cv2.putText(frame, "q = quit", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if current_letter:
            cv2.putText(frame, f"Detected: {current_letter}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("ASL Fingerspelling Recognizer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    run()