import pickle

def write_data_to_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
