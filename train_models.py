import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load cleaned articles: 0 = real, 1 = fake
news = pd.read_csv("data/news_cleaned.csv")
X = news["text"]
y = news["label"]

# Use the same training and testing articles for both models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

# Convert article text into numeric features
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Initialize the two models
logistic = LogisticRegression(max_iter=1000, random_state=123)
tree = DecisionTreeClassifier(max_depth=10, random_state=123)

# Train, predict, and evaluate each model
models = [
    ("Logistic Regression", logistic),
    ("Decision Tree", tree)
]

for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n" + name)
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, predictions, zero_division=0):.4f}")
