# Snorkel Weak Supervision Pipeline — Amazon Product Review Sentiment Classification

A lab assignment implementation of Snorkel's weak supervision and data augmentation pipeline, applied to Amazon product reviews. The goal is to classify reviews as **positive** or **negative** without relying on hand-labeled training data.

Original - https://github.com/raminmohammadi/MLOps/tree/main/Labs/Data_Labs/Data_Labeling_Labs

I have created my project as an adaptation of both file 01 and 02

---

## What This Project Does

This notebook replicates the core ideas from the Snorkel introductory tutorials (Data Labeling + Data Augmentation) on a new use case — Amazon electronics reviews. Instead of manually labeling thousands of reviews, we use programmatic labeling functions to weakly supervise a classifier.

The full pipeline looks like this:

```
Raw Amazon Reviews (no training labels)
        ↓
8 Labeling Functions (keywords, regex, heuristics, TextBlob)
        ↓
LabelModel (automatically weighted combination of LF votes)
        ↓
Logistic Regression Classifier trained on weak labels
        ↓
Data Augmentation using Synonym Replacement TFs
        ↓
Final Classifier → 84.0% Test Accuracy
```

---

## Results

| Step                          | Accuracy |
|-------------------------------|----------|
| Majority Vote Baseline        | 71.7%    |
| LabelModel                    | 73.9%    |
| Classifier (original data)    | 76.5%    |
| Classifier (augmented data)   | 84.0%    |

---

## Project Structure

```
├── productReviews.ipynb        # Main Jupyter notebook with full pipeline
└── README.md                   # This file
```

---

## Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher
- pip
- Jupyter Notebook or VSCode with Jupyter extension

### 2. Install Dependencies

Run the following in your terminal or in a notebook cell:

```bash
pip install snorkel datasets textblob scikit-learn pandas numpy spacy
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

### 3. Fix SSL Certificate Issue (Mac only)

If you are on a Mac and encounter SSL errors when downloading NLTK data (as I initially did), run this in your terminal:

```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

Replace `3.13` with your Python version if different.

### 4. Download NLTK Data

Run this in your notebook:

```python
import nltk
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
```

### 5. Run the Notebook

Open `notebook.ipynb` and run all cells from top to bottom.

---

## Pipeline Walkthrough

### Step 1 — Load Dataset
We use the `amazon_polarity` dataset from Hugging Face, sampling 4,000 balanced training reviews (2,000 positive, 2,000 negative) and 1,000 test reviews. Review title and content are combined into a single text field.

### Step 2 — Define Label Constants
```python
ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1
```

### Step 3 — Write Labeling Functions (LFs)
We write 8 LFs that programmatically assign labels to reviews:

| LF | Type | Signal |
|----|------|--------|
| `lf_positive_words` | Keyword | Words like "great", "amazing", "love" |
| `lf_negative_words` | Keyword | Words like "terrible", "broken", "waste" |
| `lf_star_rating` | Regex | Phrases like "5 stars", "1 star" |
| `lf_negative_phrases` | Keyword | Phrases like "do not buy", "waste of money" |
| `lf_positive_phrases` | Keyword | Phrases like "highly recommend", "works great" |
| `lf_exclamation` | Heuristic | 3+ exclamation marks = positive |
| `lf_all_caps` | Heuristic | 2+ all-caps words = negative |
| `lf_textblob_polarity` | Third-party model | TextBlob polarity score |

These LFs achieved **74.1% coverage** of the training set.

### Step 4 — Train the LabelModel
Snorkel's `LabelModel` combines the noisy LF votes into a single probabilistic label per review, automatically estimating how much to trust each LF. Most trusted LFs were `lf_negative_words` and `lf_textblob_polarity` (weight = 1.0).

### Step 5 — Train a Classifier
A Logistic Regression model is trained on the LabelModel's probabilistic labels using bag-of-bigrams features. This model generalizes beyond the LFs to reviews they never covered.

### Step 6 — Data Augmentation
Three Transformation Functions (TFs) are applied to generate new training examples by replacing adjectives, verbs, and nouns with WordNet synonyms. The training set grew from **2,964 to 6,672 reviews**, boosting final accuracy from 76.5% to **84.0%**.

---

## Key Concepts

- **Labeling Functions (LFs)**: Noisy, programmatic rules that assign labels to unlabeled data. They don't need to be perfect.
- **LabelModel**: A model that learns to combine LF outputs by estimating their accuracy and correlations — without any ground truth labels.
- **Transformation Functions (TFs)**: Class-preserving transformations applied to existing data points to generate new training examples.
- **Weak Supervision**: The overall paradigm of using imperfect, programmatic signals instead of hand-labeled data to train ML models.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| snorkel | 0.10.0 | Weak supervision framework |
| datasets | latest | Loading Amazon Polarity dataset |
| textblob | latest | Sentiment preprocessing for LFs |
| scikit-learn | latest | Logistic Regression classifier |
| spacy | latest | NLP preprocessing for TFs |
| nltk | latest | WordNet synonyms for augmentation |
| pandas | latest | Data manipulation |
| numpy | latest | Numerical operations |

---

## References

- [Snorkel Documentation](https://snorkel.readthedocs.io/)
- [Amazon Polarity Dataset on Hugging Face](https://huggingface.co/datasets/amazon_polarity)
- [Data Augmentation in NLP](https://arxiv.org/pdf/1901.11196.pdf)
- [Prof Ramin's repository](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Data_Labs/Data_Labeling_Labs)