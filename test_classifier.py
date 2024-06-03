import cv2 as opencv
from camera_utils import initialize_camera, display_frame, check_for_quit_key
from hand_landmarks_utils import mp_hands, process_frame, detect_and_draw_hand_landmarks, extract_landmarks
from data_utils import load_model

def main(model_filename='hand_gesture_model.pkl'):
    capture = initialize_camera()
    hands_model = mp_hands.Hands(max_num_hands=1)
    model = load_model(model_filename)

    while True:
        success, frame = capture.read()
        if not success:
            break

        result = process_frame(frame, hands_model)
        detect_and_draw_hand_landmarks(result, frame)

        landmarks = extract_landmarks(result, [], None)
        if landmarks is not None:
            prediction = model.predict([landmarks])
            opencv.putText(frame, f"Gesture: {prediction[0]}", (10, 50), opencv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, opencv.LINE_AA)

        display_frame("Hand Gesture Recognition", frame)

        if check_for_quit_key('q'):
            break

    capture.release()
    opencv.destroyAllWindows()

if __name__ == "__main__":
    main()
