import cv2 as opencv
import numpy as np
from hand_landmarks_utils import mp_hands, process_frame, extract_landmarks_sequence, get_bounding_box_with_padding, extract_features
from data_utils import load_model
import os

def predict_sign(model, landmarks, image_width, image_height):
    features = extract_features(landmarks, image_width, image_height)
    features = features.reshape(1, -1)
    if features.shape[1] != model.n_features_in_:
        raise ValueError(f"Expected {model.n_features_in_} features, but got {features.shape[1]} features")
    prediction = model.predict(features)
    confidence = max(model.predict_proba(features)[0])
    return prediction[0], confidence

def main(model_filename='random_forest_model.pkl', image_width=600, image_height=500):
    print("Loading the trained model...")
    model = load_model(model_filename)

    print("Initializing camera...")
    capture = opencv.VideoCapture(0)
    capture.set(opencv.CAP_PROP_FRAME_WIDTH, image_width)
    capture.set(opencv.CAP_PROP_FRAME_HEIGHT, image_height)

    hands_model = mp_hands.Hands(max_num_hands=1)

    print("Starting video stream...")
    while True:
        success, frame = capture.read()
        if not success:
            break

        result = process_frame(frame, hands_model)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = extract_landmarks_sequence(result)
                if landmarks is not None:
                    try:
                        sign, confidence = predict_sign(model, landmarks, image_width, image_height)
                        x_min, y_min, x_max, y_max = get_bounding_box_with_padding(hand_landmarks, frame.shape)
                        opencv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        text = f'Sign: {sign}, Confidence: {confidence:.2f}'
                        opencv.putText(frame, text, (x_min, y_min - 10), opencv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    except ValueError as e:
                        print(f"Error: {e}")

        opencv.imshow("Sign Language Recognition", frame)

        if opencv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    opencv.destroyAllWindows()
    print("Video stream ended.")

if __name__ == "__main__":
    main()
