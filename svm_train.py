import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from data_utils import write_data_to_pickle

def load_normalized_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    features = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    
    return features, labels

def main(input_pickle='normalized_data.pkl', model_output='svm_model.pkl'):
    print("Loading normalized data...")
    features, labels = load_normalized_data(input_pickle)
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    print("Training SVM Classifier...")
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    print("Saving the trained model...")
    with open(model_output, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model saved to {model_output}")

if __name__ == "__main__":
    main()
