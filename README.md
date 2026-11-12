# Fake News Detection

CS4210 group project comparing logistic regression and a decision tree classifier using article text from the ISOT Fake News Dataset.

## Setup

Download and extract the dataset from:
https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

Create a `data` folder in this project and place `True.csv` and `Fake.csv` inside it.

Create and activate a Python environment, then install the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas scikit-learn
```

## Run

Run these commands from the project folder:

```bash
python inspect_data.py
python train_models.py
```

`inspect_data.py` inspects the original files, assigns labels, removes blank and exact duplicate article texts, and saves `data/news_cleaned.csv`.

`train_models.py` uses an 80/20 stratified split, converts article text into TF-IDF features, trains both models, and prints evaluation results.

Labels: `0 = real`, `1 = fake`.

## Limitations

Results measure performance against the dataset labels, not independent fact verification. Initial inspection found that the decision tree relied heavily on the word “reuters,” suggesting source-related shortcuts.

Dataset files and the Python environment are excluded from this repository.
