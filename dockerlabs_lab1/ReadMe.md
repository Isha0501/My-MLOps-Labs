# MY IMPLEMENTATION OF LAB 1: Docker Containerization 

This lab demonstrates containerizing a machine learning workflow using Docker. The original lab used the Iris dataset with a Random Forest classifier. I have modified it to use the Handwritten Digits dataset and added comprehensive evaluation metrics.

## Modifications made:

### 1. Dataset
- Original was a Iris flower dataset from sklearn.datasets
- New is a handwritten digits dataset from sklearn.datasets

### 2. Evaluation Metrics
- Original simply trained and saved the model
- New not only trains the Random Forest model, but also prints performance metrics including Accuracy, Precision, Recall and F1 score

## Machine Learning Pipeline:

### 1: Data Loading
Loads the sklearn digits dataset containing 1,797 handwritten digit images.

### 2: Data Splitting
- Training set: 80% (1,437 samples)
- Testing set: 20% (360 samples)
- Random state: 42 (for reproducibility)

### 3: Model Training
- Algorithm: Random Forest Classifier
- Parameters:
  - n_estimators: 100 trees
  - random_state: 42

### Step 4: Model Evaluation
Calculates four key metrics:
- Accuracy: Percentage of correct predictions overall
- Precision: Quality of positive predictions (weighted across all classes)
- Recall: Ability to find all positive instances (weighted across all classes)
- F1-Score: Harmonic mean of precision and recall

### Step 5: Model Persistence
Saves the trained model as `digits_model.pkl` using joblib.

## Steps to run :

docker build -t lab1:v1 .
docker run lab1:v1

(optional - takes too much space - docker save and load)


## Sample Output:

Handwritten Digits Classification Model Training

Loading Handwritten Digits dataset...
Dataset loaded: 1797 samples, 64 features

Splitting data into train and test sets...
Training samples: 1437
Testing samples: 360

Training Random Forest classifier...
Model training completed

Evaluating model performance...


MODEL EVALUATION RESULTS :


Accuracy:  0.9722 (97.22%)
Precision: 0.9726
Recall:    0.9722
F1-Score:  0.9722

Saving model...
Model saved as 'digits_model.pkl'