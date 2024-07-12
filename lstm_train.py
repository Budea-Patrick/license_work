import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
from load_symbols import load_symbols
from data_utils import load_data

def load_normalized_data(filename):
    data = load_data(filename)
    features = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    
    return features, labels

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5)) 
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main(input='normalized_augmented_data.pkl', model_output='lstm_model.h5', eval_output='lstm_eval.txt', num_classes=26):
    print("Loading normalized data...")
    features, labels = load_normalized_data(input)
    
    # Encode labels as integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    print("Normalizing the data...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    features = features.reshape(features.shape[0], 1, features.shape[1])

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    print("Building the LSTM model...")
    model = build_model((X_train.shape[1], X_train.shape[2]), num_classes)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Training the model...")
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    print("Evaluating the model on the test set...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    y_test_cat = label_encoder.inverse_transform(y_test)
    y_pred_cat = label_encoder.inverse_transform(y_pred)
    
    with open(eval_output, 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_report(y_test_cat, y_pred_cat) + "\n\n")
        
        accuracy = accuracy_score(y_test_cat, y_pred_cat)
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
        
        f.write(f"\nTraining Time: {training_time:.2f} seconds\n")
    
    print(f"Evaluation metrics and training time saved to {eval_output}")
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plotting loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    print("Saving the trained model...")
    model.save(model_output)
    print(f"Model saved to {model_output}")

if __name__ == "__main__":
    main()
