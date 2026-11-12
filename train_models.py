import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the cleaned dataset
news = pd.read_csv("data/news_cleaned.csv")

# Select the input and target
# 0 = real, 1 = fake
X = news["text"]
y = news["label"]

# Create one split for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

# Inspect the split
print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))

print("\nTraining label proportions:")
print(y_train.value_counts(normalize=True))

print("\nTesting label proportions:")
print(y_test.value_counts(normalize=True))

overlap = set(X_train).intersection(set(X_test))
print("\nArticle texts in both groups:", len(overlap))

# Convert text into numeric features
# Learn vocabulary and weights from training text only
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)

# Initialize the models
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=123
)

tree_model = DecisionTreeClassifier(
    max_depth=10,
    random_state=123
)

# Train both models on the same training features
logistic_model.fit(X_train_tfidf, y_train)
tree_model.fit(X_train_tfidf, y_train)

# Predict labels for the same test articles
logistic_predictions = logistic_model.predict(X_test_tfidf)
tree_predictions = tree_model.predict(X_test_tfidf)

# Evaluate logistic regression
print("\nLOGISTIC REGRESSION")

print("Training accuracy:", logistic_model.score(X_train_tfidf, y_train))
print("Testing accuracy:", accuracy_score(y_test, logistic_predictions))

print("\nClassification report:")
print(classification_report(
    y_test,
    logistic_predictions,
    labels=[0, 1],
    target_names=["Real", "Fake"],
    zero_division=0
))

print("Confusion matrix — rows: actual, columns: predicted [Real, Fake]")
print(confusion_matrix(y_test, logistic_predictions, labels=[0, 1]))

# Evaluate the decision tree
print("\nDECISION TREE")

print("Training accuracy:", tree_model.score(X_train_tfidf, y_train))
print("Testing accuracy:", accuracy_score(y_test, tree_predictions))

print("\nClassification report:")
print(classification_report(
    y_test,
    tree_predictions,
    labels=[0, 1],
    target_names=["Real", "Fake"],
    zero_division=0
))

print("Confusion matrix — rows: actual, columns: predicted [Real, Fake]")
print(confusion_matrix(y_test, tree_predictions, labels=[0, 1]))

# Show the decision tree's most important word features
words = vectorizer.get_feature_names_out()

word_importance = pd.DataFrame({
    "word": words,
    "importance": tree_model.feature_importances_
})

word_importance = word_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nDECISION TREE — TOP 15 WORD FEATURES")
print(word_importance.head(15).to_string(index=False))


# Collect test articles and both models' predictions
test_results = pd.DataFrame({
    "text": X_test,
    "actual": y_test,
    "logistic_prediction": logistic_predictions,
    "tree_prediction": tree_predictions
})

# Include titles to help identify the articles
test_results["title"] = news.loc[test_results.index, "title"]


# Display three mistakes from each model
prediction_columns = [
    "logistic_prediction",
    "tree_prediction"
]

for column in prediction_columns:
    mistakes = test_results[
        test_results["actual"] != test_results[column]
    ]

    print("\nMISTAKES FOR:", column)
    print("Labels: 0 = real, 1 = fake")

    for index, row in mistakes.head(3).iterrows():
        print("\nTitle:", row["title"])
        print("Actual:", row["actual"])
        print("Predicted:", row[column])
        print("Text excerpt:", row["text"][:500])
