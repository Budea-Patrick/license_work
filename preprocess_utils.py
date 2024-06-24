import numpy as np
from hand_landmarks_utils import normalize_coordinates, calculate_distances, calculate_angles

def preprocess_data(data, image_width, image_height):
    normalized_data = []
    for landmarks, label in data:
        normalized_landmarks = normalize_coordinates(landmarks, image_width, image_height)
        distances = calculate_distances(landmarks)
        angles = calculate_angles(landmarks)
        features = np.concatenate([normalized_landmarks])
        normalized_data.append((features, label))
    return normalized_data
