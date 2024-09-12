from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def train_svm(X_train, X_test, y_train, y_test):
    # Initialize the SVM model
    svm_model = SVC(kernel='linear')
    
    # Train the model
    svm_model.fit(X_train, y_train.argmax(axis=1))
    
    # Save the trained model
    joblib.dump(svm_model, 'models/svm_model.pkl')
    
    # Predict on the test data
    y_pred = svm_model.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)
    print("SVM Accuracy: {:.2f}%".format(accuracy * 100))
