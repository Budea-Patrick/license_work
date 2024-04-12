import os
import cv2 as opencv

def create_dataset_directory(dataset_directory):
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

def create_class_directories(dataset_directory, number_of_classes):
    for current_class in range(number_of_classes):
        class_directory = os.path.join(dataset_directory, str(current_class))
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)

def collect_images(capture, dataset_directory, number_of_classes, data_per_class):
    for current_class in range(number_of_classes):
        print('Collecting for class', current_class)
        while True:
            success, frame = capture.read()
            opencv.imshow('current_frame', frame)
            key = opencv.waitKey(10)
            if key == ord('a'):
                break
            elif key == ord('q'):
                return

        current_image_number = 0
        while current_image_number < data_per_class:
            success, frame = capture.read()
            opencv.imshow('current_frame', frame)
            key = opencv.waitKey(10)
            if key == ord('q'):
                return
            image_path = os.path.join(dataset_directory, str(current_class), '{}.jpg'.format(current_image_number))
            opencv.imwrite(image_path, frame)
            current_image_number += 1

def main():
    DATASET_DIRECTORY = './dataset'
    NUMBER_OF_CLASSES = 3
    DATA_PER_CLASS = 100

    create_dataset_directory(DATASET_DIRECTORY)
    create_class_directories(DATASET_DIRECTORY, NUMBER_OF_CLASSES)

    capture = opencv.VideoCapture(0)
    collect_images(capture, DATASET_DIRECTORY, NUMBER_OF_CLASSES, DATA_PER_CLASS)
    capture.release()
    opencv.destroyAllWindows()

if __name__ == "__main__":
    main()
