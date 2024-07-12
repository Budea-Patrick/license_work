import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from data_utils import load_data, write_data
import time

def load_normalized_data(filename):
    data = load_data(filename)
    
    features = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    
    return features, labels

def main(input='normalized_augmented_data.pkl', model_output='random_forest_model.pkl', eval_output='random_forest_eval.txt'):
    print("Loading normalized data...")
    features, labels = load_normalized_data(input)
    
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    min_samples_split=2,
    bootstrap=True,
    random_state=42
    )

    print("Training the RandomForestClassifier...")
    start_time = time.time()
    rf_classifier.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    y_pred = rf_classifier.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
    _, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Feature importance plot
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    
    # metrics
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    with open(eval_output, 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred) + "\n\n")
        
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Weighted Precision: {precision:.2f}\n")
        f.write(f"Weighted Recall: {recall:.2f}\n")
        f.write(f"Weighted F1-score: {fscore:.2f}\n")
        f.write(f"Support per class: {support}\n")
        
        f.write(f"\nTraining Time: {training_time:.2f} seconds\n")
    print(f"Evaluation metrics and training time saved to {eval_output}")
    
    write_data(model_output, rf_classifier)
    print(f"Model saved to {model_output}")

if __name__ == "__main__":
    main()
