import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_gcs_path", required=True, help="GCS path to training CSV")
    parser.add_argument("--model_output_gcs", required=True, help="GCS path to save model")
    args = parser.parse_args()

    # --- Load Data ---
    print(f"Loading data from: {args.data_gcs_path}")
    df = pd.read_csv(args.data_gcs_path)
    print(f"Dataset shape: {df.shape}")

    # --- Drop irrelevant/high-cardinality columns ---
    df = df.drop(columns=["Id", "CITY", "STATE", "Profession"])

    # --- Encode categorical columns ---
    df = pd.get_dummies(df, columns=["Married.Single", "House_Ownership", "Car_Ownership"])

    # --- Split features and target ---
    X = df.drop(columns=["Risk_Flag"])
    y = df["Risk_Flag"]

    print(f"Class distribution:\n{y.value_counts()}")

    # --- Train/Test Split (stratified for class imbalance) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Build Pipeline: Scaler + Gradient Boosting ---
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ))
    ])

    # --- Train ---
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # --- Save model locally ---
    local_model_path = "model.joblib"
    joblib.dump(pipeline, local_model_path)
    print(f"Model saved locally as {local_model_path}")

    # --- Upload model to GCS ---
    gcs_model_path = os.path.join(args.model_output_gcs, local_model_path)
    os.system(f"gcloud storage cp {local_model_path} {gcs_model_path}")
    print(f"Model uploaded to: {gcs_model_path}")

if __name__ == "__main__":
    main()