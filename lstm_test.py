import numpy as np
import cv2 as opencv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from hand_landmarks_utils import mp_hands, process_frame, extract_landmarks_sequence, extract_features, get_bounding_box_with_padding
from data_utils import load_data
from camera_utils import initialize_camera, display_frame, check_for_quit_key
from load_symbols import load_symbols
import time

def load_normalized_data(filename):
    data = load_data(filename)
    features = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    return features, labels

def predict_sign(model, landmarks, scaler, image_width, image_height, label_map):
    features = extract_features(landmarks, image_width, image_height)
    features = scaler.transform([features])
    features = features.reshape(features.shape[0], 1, features.shape[1])
    
    start_time = time.time()  # Start timing
    prediction = model.predict(features)
    end_time = time.time()  # End timing
    
    sign_index = np.argmax(prediction)
    sign = label_map[sign_index]
    confidence = np.max(prediction)
    latency = end_time - start_time
    return sign, confidence, latency

def main(model_filename='lstm_model.h5', input_pickle='normalized_augmented_data.pkl', image_width=1000, image_height=800, quit_key='q'):
    print("Loading the trained model...")
    model = load_model(model_filename)

    print("Loading normalized data for scaler fitting...")
    features, _ = load_normalized_data(input_pickle)

    print("Fitting scaler on the data...")
    scaler = StandardScaler()
    scaler.fit(features)

    print("Loading symbols...")
    symbols = load_symbols('symbols.meta')
    classes = list(symbols.values())

    print("Initializing camera...")
    capture = initialize_camera()

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
                        sign, confidence, latency = predict_sign(model, landmarks, scaler, image_width, image_height, classes)
                        x_min, y_min, x_max, y_max = get_bounding_box_with_padding(hand_landmarks, frame.shape)
                        opencv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        text = f'Sign: {sign}, Confidence: {confidence:.2f}, Latency: {latency*1000:.2f} ms'
                        opencv.putText(frame, text, (x_min, y_min - 10), opencv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    except ValueError as e:
                        print(f"Error: {e}")

        display_frame("Sign Language Recognition", frame)

        if check_for_quit_key(quit_key):
            break

    capture.release()
    opencv.destroyAllWindows()
    print("Video stream ended.")

if __name__ == "__main__":
    main()
