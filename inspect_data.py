import pandas as pd

real = pd.read_csv("data/True.csv")
fake = pd.read_csv("data/Fake.csv")

print("Real dataset shape:", real.shape)
print("Fake dataset shape:", fake.shape)

print("\nReal columns:", real.columns.tolist())
print("Fake columns:", fake.columns.tolist())

print("\nExample article:")
print(real.head(1))

# Check for missing values in each column
print("\nMissing values — real:")
print(real.isna().sum())

print("\nMissing values — fake:")
print(fake.isna().sum())

# Check for empty or whitespace-only article text
print("\nBlank text — real:",
      real["text"].fillna("").str.strip().eq("").sum())

print("Blank text — fake:",
      fake["text"].fillna("").str.strip().eq("").sum())

# Count repeated article texts within each file
print("\nDuplicate text — real:", real.duplicated(subset=["text"]).sum())
print("Duplicate text — fake:", fake.duplicated(subset=["text"]).sum())

# Assign the target labels
real["label"] = 0
fake["label"] = 1

# Combine both tables
news = pd.concat([real, fake], ignore_index=True)

print("\nCombined dataset shape:", news.shape)
print("\nLabel counts:")
print(news["label"].value_counts())

# Keep rows whose article text contains something besides whitespace
news = news[news["text"].str.strip() != ""].copy()

print("\nRows after removing blank text:", len(news))

# Count how many distinct labels each exact article text has
labels_per_text = news.groupby("text")["label"].nunique()

print("Texts with conflicting labels:", (labels_per_text > 1).sum())

# Keep the first occurrence of each exact article text
before = len(news)

news = news.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("\nDuplicate rows removed:", before - len(news))
print("Final dataset shape:", news.shape)

print("\nFinal label counts:")
print(news["label"].value_counts())

# Remove exact duplicate article texts
before = len(news)

news = news.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("\nDuplicate rows removed:", before - len(news))
print("Final dataset shape:", news.shape)

# Check the final class counts
print("\nFinal label counts (0 = real, 1 = fake):")
print(news["label"].value_counts())

# Verify that cleaning worked
assert news["text"].notna().all(), "Missing article text remains"
assert news["text"].str.strip().ne("").all(), "Blank article text remains"
assert not news["text"].duplicated().any(), "Duplicate article text remains"
assert set(news["label"].unique()) == {0, 1}, "Unexpected labels"

# Save the prepared dataset without adding a row-number column
output_path = "data/news_cleaned.csv"
news.to_csv(output_path, index=False)

print("\nAll checks passed.")
print("Saved cleaned dataset to:", output_path)
print("Original True.csv and Fake.csv files are unchanged.")
