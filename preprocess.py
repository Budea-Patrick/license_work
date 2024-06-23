import pickle
from preprocess_utils import preprocess_data
from data_utils import write_data_to_pickle

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def main(input_pickle='training_data.pkl', output_pickle='normalized_data.pkl', image_width=600, image_height=500):
    print("Loading data...")
    data = load_data(input_pickle)
    print("Normalizing data...")
    normalized_data = preprocess_data(data, image_width, image_height)
    print("Saving normalized data...")
    write_data_to_pickle(output_pickle, normalized_data)
    print(f"Preprocessing complete. Normalized data saved to {output_pickle}")

if __name__ == "__main__":
    main()
