import cv2 as opencv
from camera_utils import initialize_camera, display_frame
from hand_landmarks_utils import mp_hands, process_frame, detect_and_draw_hand_landmarks, extract_landmarks_sequence
from load_symbols import load_symbols
import os
import time
import pickle

def start_data_extraction(current_class):
    print(f"Started data extraction for class {current_class}")
    start_time = time.time()
    return True, start_time

def stop_data_extraction(current_class_index, classes, start_key):
    current_class_index = (current_class_index + 1) % len(classes)
    if current_class_index == 0:
        return False, current_class_index, None
    current_class = classes[current_class_index]
    print(f"Stopped extraction. Press '{start_key.upper()}' to start extraction for class {current_class}")
    return False, current_class_index, current_class

def handle_extraction(current_class, frame, hands_model, data, start_time):
    elapsed_time = time.time() - start_time
    result = process_frame(frame, hands_model)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = extract_landmarks_sequence(result)
            if landmarks is not None:
                data.append((landmarks, current_class))
    if elapsed_time >= 15:
        return data, False
    return data, True

def main(quit_key='q', start_key='a', output_file='hand_landmarks_data.pkl'):
    capture = initialize_camera()
    hands_model = mp_hands.Hands(max_num_hands=1)
    extracting = False
    
    symbols = load_symbols('symbols.meta')
    classes = list(symbols.values())
    
    current_class_index = 0
    current_class = classes[current_class_index]

    data = []
    start_time = None

    print(f"Press '{start_key.upper()}' to start data extraction for class {current_class}")
    print("Press 'Q' to quit")

    while True:
        success, frame = capture.read()
        if not success:
            break

        if extracting:
            data, extracting = handle_extraction(current_class, frame, hands_model, data, start_time)
            if not extracting:
                extracting, current_class_index, current_class = stop_data_extraction(current_class_index, classes, start_key)
                if current_class is None:
                    break

        result = process_frame(frame, hands_model)
        detect_and_draw_hand_landmarks(result, frame)
        display_frame("my image", frame)

        key = opencv.waitKey(1)
        if key == ord(start_key) and not extracting:
            extracting, start_time = start_data_extraction(current_class)
        if key == ord(quit_key):
            break

    capture.release()
    opencv.destroyAllWindows()

    print("Saving collected data...")
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {output_file}")
    print("Data collection complete")

if __name__ == "__main__":
    main()
