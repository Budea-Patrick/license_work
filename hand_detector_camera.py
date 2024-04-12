import cv2 as opencv
import mediapipe as mp

def initialize_camera():
    capture = opencv.VideoCapture(0)
    capture.set(opencv.CAP_PROP_FRAME_WIDTH, 600)
    capture.set(opencv.CAP_PROP_FRAME_HEIGHT, 500)
    return capture

def process_frame(frame, hands_model):
    rgb_frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)
    result = hands_model.process(rgb_frame)
    return result

def detect_and_draw_hand_landmarks(result, frame):
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

def display_frame(window_name, frame):
    opencv.imshow(window_name, frame)

def check_for_quit_key(quit_key):
    return opencv.waitKey(1) == ord(quit_key)

def capture_and_process_frame(capture, hands_model):
    success, frame = capture.read()
    if success:
        result = process_frame(frame, hands_model)
        detect_and_draw_hand_landmarks(result, frame)
        display_frame("my image", frame)
    return success

def main(quit_key='q'):
    capture = initialize_camera()
    mp_hands = mp.solutions.hands.Hands(
        max_num_hands = 1
    )

    while capture_and_process_frame(capture, mp_hands):
        if check_for_quit_key(quit_key):
            break

    capture.release()
    opencv.destroyAllWindows()

if __name__ == "__main__":
    main()
