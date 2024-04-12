import os
import cv2 as opencv
import mediapipe as mp

INPUT_FOLDER = './dataset'
OUTPUT_FOLDER = './dataset_hands_only'

def extract_hand_from_image(image_path, output_path, expand_factor=1.5):
    image = opencv.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at '{image_path}'")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    image_rgb = opencv.cvtColor(image, opencv.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            bbox = _get_bounding_box(image, hand_landmarks, expand_factor)
            hand_roi = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            opencv.imwrite(output_path, hand_roi)
            print(f"Hand extracted and saved to '{output_path}'")
    else:
        print(f"No hand detected in '{image_path}'")

def _get_bounding_box(image, landmarks, expand_factor=1.5):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0

    for landmark in landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, x_min - int(width * (expand_factor - 1) / 2))
    x_max = min(image_width, x_max + int(width * (expand_factor - 1) / 2))
    y_min = max(0, y_min - int(height * (expand_factor - 1) / 2))
    y_max = min(image_height, y_max + int(height * (expand_factor - 1) / 2))

    return x_min, y_min, x_max, y_max

def create_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def extract_hand_from_folder(folder_path, output_folder):
    create_output_folder(output_folder)

    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            input_subfolder = os.path.join(root, dir_name)
            output_subfolder = os.path.join(output_folder, os.path.relpath(input_subfolder, folder_path))
            create_output_folder(output_subfolder)

        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_image_path = os.path.join(root, file_name)
                output_image_path = os.path.join(output_folder, os.path.relpath(input_image_path, folder_path))
                extract_hand_from_image(input_image_path, output_image_path)

def main():
    extract_hand_from_folder(INPUT_FOLDER, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
