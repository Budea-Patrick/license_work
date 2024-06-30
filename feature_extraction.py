import os
import cv2 as opencv
from hand_landmarks_utils import mp_hands, process_frame, extract_landmarks_sequence
from data_utils import write_data

def extract_features_from_image(image_path, hands_model):
    frame = opencv.imread(image_path)
    result = process_frame(frame, hands_model)
    landmarks = extract_landmarks_sequence(result)
    if landmarks is not None:
        return landmarks
    else:
        print(f"No hand detected in image: {image_path}")
        return None

def process_folder(folder_path, hands_model, class_label):
    features = []
    image_index = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image_features = extract_features_from_image(image_path, hands_model)
            if image_features is not None:
                features.append((image_features, class_label))
                image_index += 1
    return features

def main(input_base_dir='augmented_images', output_pickle='training_data.pkl'):
        
    hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence = 0.4)
    all_features = []

    for class_folder in os.listdir(input_base_dir):
        folder_path = os.path.join(input_base_dir, class_folder)
        if os.path.isdir(folder_path):
            class_label = class_folder
            print(f"Processing folder: {folder_path}")
            class_features = process_folder(folder_path, hands_model, class_label)
            all_features.extend(class_features)
    
    write_data(output_pickle, all_features)
    print(f"Feature extraction complete. Data saved to {output_pickle}")

if __name__ == "__main__":
    main()
