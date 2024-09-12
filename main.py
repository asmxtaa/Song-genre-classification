from src.data_loader import load_data
from src.feature_extraction import extract_features
from src.svm_classifier import train_svm
from src.random_forest_classifier import train_random_forest
from src.cnn_classifier import train_cnn

if __name__ == "__main__":
    # Load the dataset and labels
    X_train, X_test, y_train, y_test, label_encoder = load_data()
    
    # Train and evaluate the SVM classifier
    print("Training SVM...")
    train_svm(X_train, X_test, y_train, y_test)
    
    # Train and evaluate the Random Forest classifier
    print("Training Random Forest...")
    train_random_forest(X_train, X_test, y_train, y_test)
    
    # Train and evaluate the CNN model
    print("Training CNN...")
    train_cnn(X_train, X_test, y_train, y_test)
