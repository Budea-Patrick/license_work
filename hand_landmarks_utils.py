import mediapipe as mp
import cv2 as opencv
import numpy as np
import os

mp_hands = mp.solutions.hands

def process_frame(frame, hands_model):
    rgb_frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)
    result = hands_model.process(rgb_frame)
    return result

def extract_landmarks_sequence(result):
    landmarks_seq = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks_seq.append([lm.x, lm.y, lm.z])
    if len(landmarks_seq) == 0:
        return None
    return np.array(landmarks_seq).flatten()

def normalize_coordinates(landmarks, image_width, image_height):
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]  # Assuming the wrist is the first landmark
    landmarks[:, 0] = (landmarks[:, 0] - wrist[0]) / image_width   # Normalize x relative to wrist
    landmarks[:, 1] = (landmarks[:, 1] - wrist[1]) / image_height  # Normalize y relative to wrist
    return landmarks.flatten()

def get_bounding_box_with_padding(hand_landmarks, frame_shape, padding=50):
    h, w, _ = frame_shape
    x_min, x_max, y_min, y_max = w, 0, h, 0

    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y

    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, w)
    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, h)

    return x_min, y_min, x_max, y_max

def calculate_distances(landmarks):
    landmarks = landmarks.reshape(-1, 3)
    distances = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            distances.append(dist)
    return np.array(distances)

def calculate_angles(landmarks):
    landmarks = landmarks.reshape(-1, 3)
    angles = []
    for i in range(1, len(landmarks) - 1):
        vec1 = landmarks[i] - landmarks[i - 1]
        vec2 = landmarks[i + 1] - landmarks[i]
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angles.append(angle)
    return np.array(angles)

def extract_features(landmarks, image_width, image_height):
    normalized_landmarks = normalize_coordinates(landmarks, image_width, image_height)
    distances = calculate_distances(landmarks)
    angles = calculate_angles(landmarks)
    return np.concatenate([normalized_landmarks, distances, angles])

def detect_and_draw_hand_landmarks(result, frame):
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def save_cropped_hand_image(frame, bounding_box, image_index, output_dir):
    x_min, y_min, x_max, y_max = bounding_box
    cropped_image = frame[y_min:y_max, x_min:x_max]
    filename = os.path.join(output_dir, f"hand_{image_index}.png")
    opencv.imwrite(filename, cropped_image)