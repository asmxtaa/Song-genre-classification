import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from src.feature_extraction import extract_features

DATASET_PATH = "data/"
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_data():
    features_list = []
    labels = []
    
    # Traverse dataset folder
    for genre in GENRES:
        genre_dir = os.path.join(DATASET_PATH, genre)
        for file in os.listdir(genre_dir):
            file_path = os.path.join(genre_dir, file)
            features = extract_features(file_path)
            features_list.append(features)
            labels.append(genre)
    
    # Convert lists to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    # Encode genre labels
    label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(y)
    # print(f"y_encoded shape: {y_encoded.shape}")
    # print(f"y_encoded content: {y_encoded}")
    

    y_categorical = to_categorical(y_encoded)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder
