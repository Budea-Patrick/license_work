import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
from data_utils import load_data, write_data

def load_normalized_data(filename):
    data = load_data(filename)

    features = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    
    return features, labels

def main(input='normalized_augmented_data.pkl', model_output='svm_model.pkl', eval_output='svm_eval.txt'):
    print("Loading normalized data...")
    features, labels = load_normalized_data(input)
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': [1, 0.1, 0.01, 0.001],
        'svm__kernel': ['rbf', 'linear']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
    
    print("Training SVM Classifier with hyperparameter tuning...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    
    print("Evaluating the model...")
    y_pred = best_model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # metrics
    with open(eval_output, 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred) + "\n\n")
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"\nTraining Time: {training_time:.2f} seconds\n")
    print(f"Evaluation metrics and training time saved to {eval_output}")
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
    _, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    write_data(model_output, best_model)
    print(f"Model saved to {model_output}")

if __name__ == "__main__":
    main()
