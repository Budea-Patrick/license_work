import os
import pickle
import mediapipe as mp
import cv2

def initialize_hands(static_image_mode=True, min_detection_confidence=0.3):
    return mp.solutions.hands.Hands(static_image_mode=static_image_mode, min_detection_confidence=min_detection_confidence)

def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
    return img

def process_image(img, hands_model):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return hands_model.process(img_rgb)

def extract_landmarks(results):
    data_aux = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)
    return data_aux

def process_directory(directory_path, hands_model):
    data = []
    labels = []
    for dir_ in os.listdir(directory_path):
        dir_path = os.path.join(directory_path, dir_)
        if os.path.isdir(dir_path):
            for img_path in os.listdir(dir_path):
                img_full_path = os.path.join(dir_path, img_path)
                img = read_image(img_full_path)
                if img is not None:
                    results = process_image(img, hands_model)
                    data_aux = extract_landmarks(results)
                    if len(data_aux) == 42:
                        print(f"Hands detected in image: {img_full_path}")
                        data.append(data_aux)
                        labels.append(dir_)
                    else:
                        print(f"Unexpected number of landmarks in image: {img_full_path}")
    return data, labels

def save_data(data, labels, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump({'data': data, 'labels': labels}, file)

def main(data_dir, output_file):
    hands_model = initialize_hands()
    data, labels = process_directory(data_dir, hands_model)
    save_data(data, labels, output_file)
    print("Data generation completed.")
    print(f"Number of samples: {len(data)}")

if __name__ == "__main__":
    DATA_DIR = "./dataset"
    OUTPUT_FILE = 'data.pickle'
    main(DATA_DIR, OUTPUT_FILE)
