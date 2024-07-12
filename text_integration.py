import cv2 as opencv
from hand_landmarks_utils import mp_hands, process_frame, extract_landmarks_sequence, extract_features
from data_utils import load_data
from load_symbols import load_symbols
from camera_utils import initialize_camera
import pyautogui
import time

def predict_sign(model, landmarks, image_width, image_height):
    features = extract_features(landmarks, image_width, image_height)
    features = features.reshape(1, -1)
    if features.shape[1] != model.n_features_in_:
        raise ValueError(f"Expected {model.n_features_in_} features, got {features.shape[1]}")
    prediction = model.predict(features)
    return prediction[0]

def perform_action(sign, symbols):
    if sign in symbols.values():
        pyautogui.typewrite(sign.lower())
            

def main(model_filename='svm_model.pkl', image_width=1000, image_height=800):
    print("Loading the trained model...")
    model = load_data(model_filename)
    
    print("Loading symbols...")
    symbols = load_symbols('symbols.meta')

    print("Initializing camera...")
    capture = initialize_camera()
    hands_model = mp_hands.Hands(max_num_hands=1)

    start_time = time.time()
    prediction_interval = 1.5  # Predict every 1 second

    print("Starting video stream...")
    while True:
        success, frame = capture.read()
        if not success:
            break
        if time.time() - start_time >= prediction_interval:
            result = process_frame(frame, hands_model)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = extract_landmarks_sequence(result)
                    if landmarks is not None:
                        try:
                            sign = predict_sign(model, landmarks, image_width, image_height)
                            print(f"Predicted sign: {sign}")
                            perform_action(sign, symbols)
                        except ValueError as e:
                            print(f"Error: {e}")
            start_time = time.time()

        opencv.imshow("Sign Language Recognition", frame)
        if opencv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    opencv.destroyAllWindows()
    print("Video stream ended.")

if __name__ == "__main__":
    main()
