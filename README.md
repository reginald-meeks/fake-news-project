# Fake News Detection

CS4210 group project comparing logistic regression and a decision tree classifier on the ISOT Fake News Dataset.

## Setup

Download the dataset:
https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

Create a `data` folder and place `True.csv` and `Fake.csv` inside it.

From the project folder, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas scikit-learn
```

## Run

```bash
python inspect_data.py
python train_models.py
```

- `inspect_data.py` labels and combines the articles, removes blank and exact duplicate texts, and saves `data/news_cleaned.csv`.
- `train_models.py` converts text into TF-IDF features, trains both models, and prints accuracy, precision, recall, and F1.

Labels are `0 = real` and `1 = fake`. Precision, recall, and F1 are reported for the fake class.

## Experiment Settings

Both models use the same 80/20 stratified training/test split with `random_state=123`.

- TF-IDF: up to 5,000 features, fitted on training text only.
- Logistic regression: `max_iter=1000`.
- Decision tree: `max_depth=10`.
- Both models use `random_state=123`.

## Data Preparation

The original dataset contained 44,898 articles. Preparation removed 631 blank-text rows and then 5,623 exact duplicate rows, leaving 38,644 articles: 21,191 real and 17,453 fake.

## Limitations

The models predict dataset labels rather than independently verify facts. An earlier diagnostic found that “reuters” accounted for approximately 98.49% of the decision tree’s impurity-based feature importance, suggesting heavy reliance on a source identifier. Exact deduplication does not remove all near-duplicates or source/style bias.

Dataset files and the virtual environment are excluded from GitHub. Each group member must download the data and follow the setup instructions.
