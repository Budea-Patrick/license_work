import os
import cv2 as opencv
from data_augmentation_utils import augment_image

def augment_dataset(input_dir, output_dir, augmentations_per_image=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_index = 0
    for class_folder in os.listdir(input_dir):
        class_folder_path = os.path.join(input_dir, class_folder)
        if os.path.isdir(class_folder_path):
            class_output_dir = os.path.join(output_dir, class_folder)
            if not os.path.exists(class_output_dir):
                os.makedirs(class_output_dir)
            
            for filename in os.listdir(class_folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_folder_path, filename)
                    image = opencv.imread(image_path)
                    
                    # Save the original image
                    output_path = os.path.join(class_output_dir, f"aug_{image_index}.png")
                    opencv.imwrite(output_path, image)
                    image_index += 1
                    
                    # Create and save augmentations
                    for i in range(augmentations_per_image):
                        augmented_image = augment_image(image)
                        output_path = os.path.join(class_output_dir, f"aug_{image_index}.png")
                        opencv.imwrite(output_path, augmented_image)
                        image_index += 1

def main(input_dir='hand_images', output_dir='augmented_images', augmentations_per_image=5):
    print(f"Augmenting dataset from {input_dir} and saving to {output_dir} with {augmentations_per_image} augmentations per image...")
    augment_dataset(input_dir, output_dir, augmentations_per_image)
    print("Data augmentation complete.")

if __name__ == "__main__":
    main()
