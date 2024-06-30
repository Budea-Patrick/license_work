import cv2 as opencv
from camera_utils import initialize_camera, display_frame
from hand_landmarks_utils import mp_hands, process_frame, detect_and_draw_hand_landmarks, get_bounding_box_with_padding, save_cropped_hand_image
from load_symbols import load_symbols
import os

def start_data_extraction(current_class):
    print(f"Started data extraction for class {current_class}")
    return True

def stop_data_extraction(current_class_index, classes, start_key):
    current_class_index = (current_class_index + 1) % len(classes)
    if current_class_index == 0:
        return False, current_class_index, None
    current_class = classes[current_class_index]
    print(f"Stopped extraction. Press '{start_key.upper()}' to start extraction for class {current_class}")
    return False, current_class_index, current_class

def handle_extraction(current_class, frame, hands_model, base_output_dir, image_index, nr_images):
    result = process_frame(frame, hands_model)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            bounding_box = get_bounding_box_with_padding(hand_landmarks, frame.shape)
            class_output_dir = os.path.join(base_output_dir, current_class)
            if not os.path.exists(class_output_dir):
                os.makedirs(class_output_dir)
            print(f"Saving image {image_index + 1} for class {current_class}")
            save_cropped_hand_image(frame, bounding_box, image_index, class_output_dir)
            x_min, y_min, x_max, y_max = bounding_box
            opencv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            image_index += 1
            if image_index >= nr_images:
                return image_index, False
    return image_index, True

def main(quit_key='q', start_key='a', base_output_dir='hand_images', nr_images = 500):
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        
    capture = initialize_camera()
    hands_model = mp_hands.Hands(max_num_hands=1)
    extracting = False
    
    symbols = load_symbols('symbols.meta')
    classes = list(symbols.values())
    
    current_class_index = 0
    current_class = classes[current_class_index]

    image_index = 0

    print(f"Press '{start_key.upper()}' to start data extraction for class {current_class}")
    print("Press 'Q' to quit")

    while True:
        success, frame = capture.read()
        if not success:
            break

        if extracting:
            image_index, extracting = handle_extraction(current_class, frame, hands_model, base_output_dir, image_index, nr_images)
            if not extracting:
                extracting, current_class_index, current_class = stop_data_extraction(current_class_index, classes, start_key)
                if current_class is None:
                    break
                image_index = 0

        result = process_frame(frame, hands_model)
        detect_and_draw_hand_landmarks(result, frame)
        display_frame("my image", frame)

        key = opencv.waitKey(1)
        if key == ord(start_key) and not extracting:
            extracting = start_data_extraction(current_class)
        if key == ord(quit_key):
            break

    capture.release()
    opencv.destroyAllWindows()
    print("Data collection complete")

if __name__ == "__main__":
    main()
