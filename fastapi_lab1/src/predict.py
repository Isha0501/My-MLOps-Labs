import joblib

CLASS_NAMES = {0: "malignant", 1: "benign"}


def predict_data(X):
    """
    Predict the class label for the input data.

    Args:
        X (list or numpy.ndarray): Input feature array of shape (1, 30).

    Returns:
        label (str): Predicted class name — either "malignant" or "benign".
        confidence (float): The model's confidence (probability) for the prediction.
    """
    model = joblib.load("../model/cancer_model.pkl")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    label = CLASS_NAMES[int(y_pred[0])]
    confidence = round(float(y_prob[0][int(y_pred[0])]), 4)
    return label, confidence