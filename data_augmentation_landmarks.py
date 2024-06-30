from data_augmentation_utils import augment_data
from preprocess_utils import preprocess_data
from data_utils import write_data_to_pickle, load_data

def main(input_pickle='hand_landmarks_data.pkl', output_pickle='normalized_augmented_data.pkl', image_width=1000, image_height=800):
    print("Loading data...")
    data = load_data(input_pickle)
    
    print("Augmenting data...")
    augmented_data = augment_data(data, image_width)
    
    print("Normalizing augmented data...")
    normalized_data = preprocess_data(augmented_data, image_width, image_height)
    
    print(f"Original data size: {len(data)}")
    print(f"Augmented data size: {len(augmented_data)}")
    
    print("Saving normalized augmented data...")
    write_data_to_pickle(output_pickle, normalized_data)
    print(f"Data augmentation and normalization complete. Data saved to {output_pickle}")

if __name__ == "__main__":
    main()
