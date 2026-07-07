# Recipe Cuisine Classification

Machine learning project for classifying recipes as **American** or **Italian** using sparse TF-IDF ingredient features.

This repository contains the modelling approach and reproducible code structure, but **does not include the original dataset** because it was provided for a university data challenge.

## Project Overview

The task was a binary classification problem using 40 non-negative TF-IDF ingredient features. The final approach combined exact-match lookup, feature engineering, and an ensemble of complementary machine learning models.

The strongest result came from a wide-pool weighted blend of six classifiers, achieving approximately **92.9% cross-validation accuracy**.

## Key Methods

- **Exact-match classification** for recipes whose feature vectors had appeared in the training data
- **Feature engineering** from raw TF-IDF values, ingredient presence indicators, and row-level recipe summaries
- **Supervised fold-based features** including rarity scores and ingredient log-odds, computed within each cross-validation fold to avoid leakage
- **Model ensemble** combining ExtraTrees, SVM, Zero-Inflated Gaussian Naive Bayes, LDA, Bernoulli Naive Bayes, and MLP classifiers
- **Out-of-fold blending** to optimize model weights and classification thresholds

## Repository Contents

```text
recipe-cuisine-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── features.py
│   ├── exact_match.py
│   ├── models.py
│   ├── train.py
│   └── predict.py
└── notebooks/
    └── model_development_summary.ipynb
```

## Data Availability

The original `train.csv`, `test.csv`, and submission files are not included in this repository because the data was provided through a university course challenge.

To run the code locally, place the required files in a local, untracked directory:

```text
data/
├── train.csv
└── test.csv
```

The expected format is:

- `train.csv`: one label column followed by 40 TF-IDF ingredient feature columns
- `test.csv`: 40 TF-IDF ingredient feature columns

## Results Summary

| Model | CV Error | CV Accuracy |
|---|---:|---:|
| OLS Linear Probability | 0.2319 | 0.768 |
| L2 Logistic Regression | 0.2025 | 0.798 |
| Linear SVM | 0.2003 | 0.800 |
| Random Forest | 0.0884 | 0.912 |
| ExactMatch + ExtraTrees | 0.0816 | 0.918 |
| Final Wide-Pool Blend | 0.0713 | 0.929 |

## Main Takeaways

Tree-based and ensemble models substantially outperformed simpler linear models, suggesting that nonlinear ingredient interactions were important for distinguishing American and Italian recipes. Exact feature-vector duplicates also played a major role, motivating an exact-match layer before fallback model prediction.

Feature interpretation showed that both ingredient presence and ingredient intensity mattered. Some ingredients had different signals depending on whether they were merely present or appeared with high TF-IDF weight.

## Requirements

```text
numpy
pandas
scikit-learn>=1.3
scipy
jupyter
matplotlib
```

## Notes

This repository is intended to document the modelling process while respecting the data-sharing restrictions of the original university challenge.
