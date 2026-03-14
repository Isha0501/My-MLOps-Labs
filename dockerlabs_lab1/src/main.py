# Import necessary libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

if __name__ == '__main__':
    print("Handwritten Digits Classification Model Training")
    
    # Load the Digits dataset
    print("\nLoading Handwritten Digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Split the data into training and testing sets
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # Train a Random Forest classifier
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed")

    # Make predictions
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print results
    print("\n")
    print("MODEL EVALUATION RESULTS :\n")
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Save the model
    print("\nSaving model...")
    joblib.dump(model, 'digits_model.pkl')
    print("Model saved as 'digits_model.pkl'")
    
