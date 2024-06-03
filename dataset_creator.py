import cv2 as opencv
from camera_utils import initialize_camera, display_frame, check_for_quit_key
from hand_landmarks_utils import mp_hands, process_frame, detect_and_draw_hand_landmarks, extract_landmarks
from data_utils import write_data_to_pickle
from load_symbols import load_symbols

def start_data_extraction(current_class):
    print(f"Started data extraction for class {current_class}")
    return True, opencv.getTickCount()

def stop_data_extraction(current_class_index, classes, start_key):
    current_class_index = (current_class_index + 1) % len(classes)
    if current_class_index == 0:
        return False, current_class_index, None
    current_class = classes[current_class_index]
    print(f"Stopped extraction. Press '{start_key.upper()}' to start extraction for class {current_class}")
    return False, current_class_index, current_class

def handle_extraction(start_time, current_class, frame, data, result):
    elapsed_time = (opencv.getTickCount() - start_time) / opencv.getTickFrequency()
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    if elapsed_time > 15:
        return False
    else:
        extract_landmarks(result, data, current_class)
        return True

def main(quit_key='q', start_key='a', pickle_filename='hand_landmarks_data.pkl'):
    capture = initialize_camera()
    hands_model = mp_hands.Hands(max_num_hands=1)
    extracting = False
    data = []
    
    symbols = load_symbols('symbols.meta')
    classes = list(symbols.values())
    
    current_class_index = 0
    current_class = classes[current_class_index]
    start_time = None

    print(f"Press '{start_key.upper()}' to start data extraction for class {current_class}")
    print("Press 'Q' to quit")

    while True:
        success, frame = capture.read()
        if not success:
            break

        if extracting:
            extracting = handle_extraction(start_time, current_class, frame, data, process_frame(frame, hands_model))
            if not extracting:
                extracting, current_class_index, current_class = stop_data_extraction(current_class_index, classes, start_key)
                if current_class is None:
                    break

        detect_and_draw_hand_landmarks(process_frame(frame, hands_model), frame)
        display_frame("my image", frame)

        key = opencv.waitKey(1)
        if key == ord(start_key) and not extracting:
            extracting, start_time = start_data_extraction(current_class)
        if key == ord(quit_key):
            break

    capture.release()
    opencv.destroyAllWindows()
    write_data_to_pickle(pickle_filename, data)
    print(f"Data saved to {pickle_filename}")

if __name__ == "__main__":
    main()
