import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from data import load_data, split_data


def fit_model(X_train, y_train):
    """
    Train a Random Forest Classifier and save the model to a file.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.

    Returns:
        model (RandomForestClassifier): The trained model.
    """
    rf_classifier = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    rf_classifier.fit(X_train, y_train)
    joblib.dump(rf_classifier, "../model/cancer_model.pkl")
    print("Model saved to ../model/cancer_model.pkl")
    return rf_classifier


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print key classification metrics.

    Args:
        model (RandomForestClassifier): The trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): True test labels.
    """
    y_pred = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (of predicted benign, how many truly are)")
    print(f"Recall   : {recall:.4f}  (of actual benign, how many we caught)")
    print("------------------------\n")


if __name__ == "__main__":
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)