import pickle
import numpy as np
import cv2 as opencv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from hand_landmarks_utils import mp_hands, process_frame, extract_landmarks_sequence, extract_features, get_bounding_box_with_padding

def load_normalized_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    features = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    
    return features, labels

def predict_sign(model, landmarks, scaler, image_width, image_height, label_map):
    features = extract_features(landmarks, image_width, image_height)
    features = scaler.transform([features])
    features = features.reshape(features.shape[0], 1, features.shape[1])  # Reshape for LSTM [samples, time steps, features]

    prediction = model.predict(features)
    sign_index = np.argmax(prediction)
    sign = label_map[sign_index]
    confidence = np.max(prediction)
    
    return sign, confidence

def main(model_filename='lstm_model.h5', input_pickle='normalized_data.pkl', image_width=600, image_height=500):
    print("Loading the trained model...")
    model = load_model(model_filename)

    print("Loading normalized data for scaler fitting...")
    features, _ = load_normalized_data(input_pickle)

    print("Fitting scaler on the data...")
    scaler = StandardScaler()
    scaler.fit(features)

    # Define the label map based on the classes used during training
    label_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

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
                        sign, confidence = predict_sign(model, landmarks, scaler, image_width, image_height, label_map)
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
