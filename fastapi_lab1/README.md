# FastAPI Lab 1 — Breast Cancer Classifier API

**Course**: Machine Learning Operations (MLOps) — Spring 2026, Northeastern University  
**Instructor**: Professor Ramin Mohammadi  
**Original Lab**: [raminmohammadi/MLOps - FastAPI_Labs](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs)

---

## What This Lab Does

This lab builds a REST API using **FastAPI** that serves a machine learning model to classify tumor measurements as either **malignant** or **benign**. The API accepts 30 numeric features derived from a digitized image of a fine needle aspirate (FNA) of a breast mass, and returns a human-readable prediction along with a confidence score.

The workflow is:
1. Load and split the Breast Cancer dataset
2. Train a Random Forest classifier and save it as a `.pkl` file
3. Serve the model through a FastAPI app with two endpoints

---

## What I Changed From the Original Lab

| Aspect | Original Lab | My Version |
|---|---|---|
| **Dataset** | Iris (4 features, 3 classes) | Breast Cancer (30 features, 2 classes) |
| **Model** | Decision Tree (`max_depth=3`) | Random Forest (100 trees, `max_depth=5`) |
| **Prediction output** | Integer (0, 1, or 2) | String label (`"malignant"` or `"benign"`) |
| **Confidence score** | Not included | Returned alongside each prediction |
| **Model evaluation** | Not included | Accuracy, Precision, and Recall printed after training |
| **Input validation** | Basic Pydantic types | `gt=0` constraints + example values on all 30 fields |
| **API metadata** | Default FastAPI title | Custom title, description, and version |

### Why These Changes?

- **Random Forest over Decision Tree**: Random Forests are more robust — they reduce overfitting by averaging across many trees, which matters for medical data.
- **Breast Cancer dataset**: More realistic than Iris for demonstrating a binary classification API. The 30 features are real clinical measurements.
- **Returning class names + confidence**: A raw integer like `0` is not meaningful to an end user or a downstream system. `"malignant"` with a confidence of `0.94` is.
- **Evaluation metrics**: Accuracy alone is misleading for medical classification. Precision and Recall give a fuller picture — especially recall, since missing a malignant tumor (false negative) is more dangerous than a false positive.

---

## File Structure

```
fastapi_lab1/
├── model/
│   └── cancer_model.pkl      # generated after running train.py
├── src/
│   ├── __init__.py
│   ├── data.py               # loads and splits the Breast Cancer dataset
│   ├── train.py              # trains Random Forest, evaluates, saves model
│   ├── predict.py            # loads model and returns label + confidence
│   └── main.py               # FastAPI app with /health and /predict endpoints
├── README.md
└── requirements.txt
```

---

## How to Run It

### 1. Prerequisites

Make sure you have Python 3.9+ installed. Check with:
```bash
python --version
```

### 2. Clone the repository

```bash
git clone https://github.com/Isha0501/My-MLOps-Labs.git
cd My-MLOps-Labs/fastapi_lab1
```

### 3. Create and activate a virtual environment

```bash
python -m venv venv

# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the model

```bash
cd src
python train.py
```

You should see output like:
```
Model saved to ../model/cancer_model.pkl

--- Model Evaluation ---
Accuracy : 0.9649
Precision: 0.9589  (of predicted benign, how many truly are)
Recall   : 0.9859  (of actual benign, how many we caught)
------------------------
```

### 6. Start the API server

```bash
uvicorn main:app --reload
```

### 7. Test the API

Open your browser and go to:
```
http://127.0.0.1:8000/docs
```

This opens the **interactive Swagger UI** where you can test both endpoints directly. The `/predict` form will be pre-filled with example values from a real malignant sample.

Alternatively, test with `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8,
    "mean_area": 1001.0, "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
    "mean_concavity": 0.3001, "mean_concave_points": 0.1471, "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871, "se_radius": 1.095, "se_texture": 0.9053,
    "se_perimeter": 8.589, "se_area": 153.4, "se_smoothness": 0.006399,
    "se_compactness": 0.04904, "se_concavity": 0.05373, "se_concave_points": 0.01587,
    "se_symmetry": 0.03003, "se_fractal_dimension": 0.006193, "worst_radius": 25.38,
    "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0,
    "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654, "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
  }'
```

Expected response:
```json
{
  "prediction": "malignant",
  "confidence": 0.9697
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns `{"status": "healthy"}` |
| `POST` | `/predict` | Accepts 30 tumor features, returns prediction + confidence |

---

## Dependencies

| Package | Version |
|---|---|
| scikit-learn | 1.5.1 |
| fastapi[all] | 0.111.1 |
| joblib | 1.4.2 |