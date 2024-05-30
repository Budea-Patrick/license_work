import cv2
import mediapipe as mp
import pickle
import numpy as np

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model_dictionary = pickle.load(file)
    return model_dictionary['model']

def initialize_camera():
    cap = cv2.VideoCapture(0)
    return cap

def initialize_hands(static_image_mode=True, min_detection_confidence=0.3):
    return mp.solutions.hands.Hands(static_image_mode=static_image_mode, min_detection_confidence=min_detection_confidence)

def process_frame(frame, hands_model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(frame_rgb)
    return results

def extract_landmarks(results):
    data_aux = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)
    return data_aux

def predict_gesture(data_aux, model, labels_dictionary):
    if len(data_aux) == 42:  # Ensure correct number of landmarks
        prediction = model.predict([np.asarray(data_aux)])
        return labels_dictionary[int(prediction[0])]
    return None

def draw_landmarks(frame, results, mp_drawing, mp_hands, mp_drawing_styles):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

def display_frame(window_name, frame):
    cv2.imshow(window_name, frame)

def main(model_path, labels_dictionary):
    model = load_model(model_path)
    cap = initialize_camera()
    hands_model = initialize_hands()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = process_frame(frame, hands_model)
        data_aux = extract_landmarks(results)
        gesture = predict_gesture(data_aux, model, labels_dictionary)
        if gesture:
            print(gesture)
        
        draw_landmarks(frame, results, mp_drawing, mp_hands, mp_drawing_styles)
        display_frame('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = './model.p'
    LABELS_DICTIONARY = {0: 'A', 1: 'B', 2: 'C'}
    main(MODEL_PATH, LABELS_DICTIONARY)
