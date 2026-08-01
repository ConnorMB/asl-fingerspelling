def landmarks_to_vector(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]
    scale = ((middle_tip.x - wrist.x) ** 2 +
             (middle_tip.y - wrist.y) ** 2 +
             (middle_tip.z - wrist.z) ** 2) ** 0.5

    vector = []
    for lm in hand_landmarks.landmark:
        vector.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale,
        ])
    return vector
