import mediapipe as mp
import cv2 as opencv
import numpy as np

mp_hands = mp.solutions.hands

def process_frame(frame, hands_model):
    rgb_frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)
    result = hands_model.process(rgb_frame)
    return result

def detect_and_draw_hand_landmarks(result, frame):
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def extract_landmarks_sequence(result, sequence_length=10):
    landmarks_seq = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks_seq.append([lm.x, lm.y, lm.z])
    if len(landmarks_seq) == 0:
        return None
    return np.array(landmarks_seq).flatten()

def update_sequence(sequence, new_data, max_length=10):
    sequence.append(new_data)
    if len(sequence) > max_length:
        sequence.pop(0)
    return sequence
