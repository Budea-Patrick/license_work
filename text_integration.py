import cv2 as opencv
import numpy as np
from hand_landmarks_utils import mp_hands, process_frame, extract_landmarks_sequence, extract_features
from data_utils import load_data
from load_symbols import load_symbols
import pyautogui
import time

def predict_sign(model, landmarks, image_width, image_height):
    features = extract_features(landmarks, image_width, image_height)
    features = features.reshape(1, -1)
    if features.shape[1] != model.n_features_in_:
        raise ValueError(f"Expected {model.n_features_in_} features, but got {features.shape[1]} features")
    prediction = model.predict(features)
    confidence = max(model.predict_proba(features)[0])
    return prediction[0]  # Return only the predicted sign, not confidence

def perform_action(sign, symbols):
    if sign in symbols.values():
        if sign == "SPACE":
            pyautogui.typewrite(" ")
        else:
            pyautogui.typewrite(sign.lower())

def main(model_filename='svm_model.pkl', image_width=1000, image_height=800):
    print("Loading the trained model...")
    model = load_data(model_filename)
    
    print("Loading symbols...")
    symbols = load_symbols('symbols.meta')
    print("Symbols loaded:", symbols.values())  # Print loaded symbols for verification

    print("Initializing camera...")
    capture = opencv.VideoCapture(0)
    capture.set(opencv.CAP_PROP_FRAME_WIDTH, image_width)
    capture.set(opencv.CAP_PROP_FRAME_HEIGHT, image_height)

    hands_model = mp_hands.Hands(max_num_hands=1)

    # Initialize timer variables
    start_time = time.time()
    prediction_interval = 1.5  # Predict every 1 second

    print("Starting video stream...")
    while True:
        success, frame = capture.read()
        if not success:
            break

        # Check if it's time to predict
        if time.time() - start_time >= prediction_interval:
            result = process_frame(frame, hands_model)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = extract_landmarks_sequence(result)
                    if landmarks is not None:
                        try:
                            sign = predict_sign(model, landmarks, image_width, image_height)
                            print(f"Predicted sign: {sign}")  # Print predicted sign for debugging
                            # Perform action based on predicted sign
                            perform_action(sign, symbols)
                            
                        except ValueError as e:
                            print(f"Error: {e}")
            
            # Reset timer for next prediction
            start_time = time.time()

        opencv.imshow("Sign Language Recognition", frame)

        if opencv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    opencv.destroyAllWindows()
    print("Video stream ended.")

if __name__ == "__main__":
    main()
