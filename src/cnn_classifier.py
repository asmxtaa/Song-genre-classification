import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape):
    model = models.Sequential()
    
    # Add convolutional layers
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # Output layer
    model.add(layers.Dense(10, activation='softmax'))  # 10 genres
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_cnn(X_train, X_test, y_train, y_test):
    # Reshape data for CNN
    X_train_cnn = X_train[:, :, np.newaxis]
    X_test_cnn = X_test[:, :, np.newaxis]
    
    # Create CNN model
    cnn_model = create_cnn_model(input_shape=(X_train_cnn.shape[1], 1))
    
    # Train the model
    cnn_model.fit(X_train_cnn, y_train, epochs=30, batch_size=32, validation_split=0.2)
    
    # Save the trained model
    cnn_model.save('models/cnn_model.h5')
    
    # Evaluate the model
    test_loss, test_acc = cnn_model.evaluate(X_test_cnn, y_test)
    print("CNN Accuracy: {:.2f}%".format(test_acc * 100))
