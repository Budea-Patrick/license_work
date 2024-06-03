import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def prepare_data(data):
    X = [landmark[:-1] for landmark in data]
    y = [landmark[-1] for landmark in data]
    return np.array(X), np.array(y)

def main(pickle_filename='hand_landmarks_data.pkl'):
    data = load_data_from_pickle(pickle_filename)
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    with open('hand_gesture_model.pkl', 'wb') as file:
        pickle.dump(clf, file)
    print("Model saved to 'hand_gesture_model.pkl'")

if __name__ == "__main__":
    main()
