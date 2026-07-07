# Recipe Cuisine Classification Code

This folder contains a privacy-safe Python reconstruction of the modelling code from the recipe cuisine classification report.

The original university-provided training and test data are intentionally not included.

## Structure

```text
src/recipe_classifier/
├── features.py       # FeatureAdder, SupervisedRecipeFeatureAdder, PresenceTransformer
├── exact_match.py    # ExactMatchClassifier
├── models.py         # Six model builders and ZeroInflatedGaussianNB
├── train.py          # OOF probabilities, weight search, submission generation
├── predict.py        # CLI wrapper for train.py
└── __init__.py
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run from the repository root after placing private data locally:

```bash
PYTHONPATH=src python -m recipe_classifier.train --train data/train.csv --test data/test.csv --out submissions
```

Expected private data format:

- `train.csv`: first column is the label, remaining columns are the 40 TF-IDF ingredient features.
- `test.csv`: 40 TF-IDF ingredient feature columns and no label column.

Do not commit `data/` or `submissions/`; both are ignored in `.gitignore`.
