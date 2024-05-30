import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(file_path):
    with open(file_path, 'rb') as file:
        data_dictionary = pickle.load(file)
    return data_dictionary['data'], data_dictionary['labels']

def check_and_fix_data_consistency(data):
    data_lengths = [len(d) for d in data]
    unique_lengths = set(data_lengths)
    if len(unique_lengths) > 1:
        print(f"Inconsistent data lengths found: {unique_lengths}")
        max_length = max(data_lengths)
        data = [np.pad(d, (0, max_length - len(d)), 'constant') if len(d) < max_length else d[:max_length] for d in data]
    return data

def prepare_data(data, labels):
    data = np.asarray(data)
    labels = np.asarray(labels)
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    return data, labels

def split_data(data, labels, test_size=0.2):
    return train_test_split(data, labels, test_size=test_size, shuffle=True, stratify=labels)

def train_model(x_train, y_train):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print(f'{score * 100:.2f}% of samples were correctly classified')
    return score

def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump({'model': model}, file)

def main(data_file_path, model_file_path):
    data, labels = load_data(data_file_path)
    data = check_and_fix_data_consistency(data)
    data, labels = prepare_data(data, labels)
    x_train, x_test, y_train, y_test = split_data(data, labels)
    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)
    save_model(model, model_file_path)

if __name__ == "__main__":
    DATA_FILE_PATH = './data.pickle'
    MODEL_FILE_PATH = './model.p'
    main(DATA_FILE_PATH, MODEL_FILE_PATH)
