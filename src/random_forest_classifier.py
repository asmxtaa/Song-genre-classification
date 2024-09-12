from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_random_forest(X_train, X_test, y_train, y_test):
    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    rf_model.fit(X_train, y_train.argmax(axis=1))
    
    # Save the trained model
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    
    # Predict on the test data
    y_pred = rf_model.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)
    print("Random Forest Accuracy: {:.2f}%".format(accuracy * 100))
