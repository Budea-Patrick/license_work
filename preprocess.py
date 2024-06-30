from preprocess_utils import preprocess_data
from data_utils import load_data, write_data

def main(input='training_data.pkl', output='normalized_data.pkl', image_width=1000, image_height=800):
    print("Loading data...")
    data = load_data(input)
    print("Normalizing data...")
    normalized_data = preprocess_data(data, image_width, image_height)
    print("Saving normalized data...")
    write_data(output, normalized_data)
    print(f"Preprocessing complete. Normalized data saved to {output}")

if __name__ == "__main__":
    main()
