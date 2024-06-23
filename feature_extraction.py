import os
import cv2 as opencv
import numpy as np
from hand_landmarks_utils import mp_hands, process_frame, extract_landmarks_sequence, get_bounding_box_with_padding, save_cropped_hand_image
from data_utils import write_data_to_pickle

def extract_features_from_image(image_path, hands_model, image_index, output_dir):
    frame = opencv.imread(image_path)
    result = process_frame(frame, hands_model)
    landmarks = extract_landmarks_sequence(result)
    if landmarks is not None:
        for hand_landmarks in result.multi_hand_landmarks:
            bounding_box = get_bounding_box_with_padding(hand_landmarks, frame.shape)
            save_cropped_hand_image(frame, bounding_box, image_index, output_dir)
        return landmarks
    else:
        print(f"No hand detected in image: {image_path}")
        return None

def process_folder(folder_path, hands_model, class_label, output_dir):
    features = []
    image_index = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image_features = extract_features_from_image(image_path, hands_model, image_index, output_dir)
            if image_features is not None:
                features.append((image_features, class_label))
                image_index += 1
    return features

def main(input_base_dir='hand_images', output_pickle='training_data.pkl', output_dir='cropped_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    all_features = []

    for class_folder in os.listdir(input_base_dir):
        folder_path = os.path.join(input_base_dir, class_folder)
        if os.path.isdir(folder_path):
            class_label = class_folder  # Assuming folder name is the class label
            print(f"Processing folder: {folder_path}")
            class_features = process_folder(folder_path, hands_model, class_label, output_dir)
            all_features.extend(class_features)
    
    write_data_to_pickle(output_pickle, all_features)
    print(f"Feature extraction complete. Data saved to {output_pickle}")

if __name__ == "__main__":
    main()
