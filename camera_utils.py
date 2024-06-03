import cv2 as opencv

def initialize_camera():
    capture = opencv.VideoCapture(0)
    capture.set(opencv.CAP_PROP_FRAME_WIDTH, 600)
    capture.set(opencv.CAP_PROP_FRAME_HEIGHT, 500)
    return capture

def display_frame(window_name, frame):
    opencv.imshow(window_name, frame)

def check_for_quit_key(quit_key):
    return opencv.waitKey(1) == ord(quit_key)
