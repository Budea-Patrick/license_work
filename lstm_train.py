import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def load_normalized_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    features = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    
    return features, labels

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))  # Increased dropout rate
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))  # Increased dropout rate
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main(input_pickle='normalized_data.pkl', model_output='lstm_model.h5', num_classes=25):
    print("Loading normalized data...")
    features, labels = load_normalized_data(input_pickle)
    
    # Encode labels as integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    print("Normalizing the data...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    features = features.reshape(features.shape[0], 1, features.shape[1])  # Reshape for LSTM [samples, time steps, features]

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    print("Building the LSTM model...")
    model = build_model((X_train.shape[1], X_train.shape[2]), num_classes)

    # Adding early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Training the model...")
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    print("Evaluating the model on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model Loss: {loss}")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    print("Saving the trained model...")
    model.save(model_output)
    print(f"Model saved to {model_output}")

if __name__ == "__main__":
    main()
