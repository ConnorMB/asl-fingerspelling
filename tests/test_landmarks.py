from landmarks import landmarks_to_vector


class FakeLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakeHandLandmarks:
    def __init__(self, points):
        self.landmark = [FakeLandmark(*p) for p in points]


def test_landmarks_to_vector_has_63_values():
    points = [(0.1, 0.1, 0.0)] * 21
    points[12] = (0.3, 0.1, 0.0)  # middle tip, distinct from wrist
    fake = FakeHandLandmarks(points)

    vector = landmarks_to_vector(fake)

    assert len(vector) == 63


def test_landmarks_to_vector_centers_wrist_at_origin():
    points = [(0.5, 0.5, 0.0)] * 21
    points[12] = (0.5, 0.3, 0.0)
    fake = FakeHandLandmarks(points)

    vector = landmarks_to_vector(fake)

    assert vector[0] == 0.0
    assert vector[1] == 0.0
    assert vector[2] == 0.0


def test_landmarks_to_vector_scales_middle_tip_to_unit_distance():
    points = [(0.5, 0.5, 0.0)] * 21
    points[12] = (0.5, 0.3, 0.0)  # 0.2 units above wrist
    fake = FakeHandLandmarks(points)

    vector = landmarks_to_vector(fake)
    mid_x, mid_y, mid_z = vector[36:39]  # landmark 12 -> indices 36,37,38
    distance = (mid_x ** 2 + mid_y ** 2 + mid_z ** 2) ** 0.5

    assert abs(distance - 1.0) < 1e-9
