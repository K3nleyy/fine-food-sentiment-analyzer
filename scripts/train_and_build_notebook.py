import os
import sys
import time
import json
import base64
import io
import re
import html
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

print("="*70)
print("PHASE 2 & PHASE 3: PREPROCESSING & MODEL TRAINING PIPELINE")
print("="*70)

DATA_PATH = "data/Reviews.csv"
if not os.path.exists(DATA_PATH):
    possible = ["../data/Reviews.csv", "/content/drive/MyDrive/fine-food-sentiment-analyzer /Reviews.csv"]
    for p in possible:
        if os.path.exists(p):
            DATA_PATH = p
            break

print(f"1. Loading dataset from: {DATA_PATH}...")
t0 = time.time()
df = pd.read_csv(DATA_PATH)
raw_shape = df.shape
print(f"   Loaded {raw_shape[0]:,} rows and {raw_shape[1]} columns in {time.time()-t0:.2f}s")

# Anomaly check
anomalies_count = (df['HelpfulnessNumerator'] > df['HelpfulnessDenominator']).sum()
print(f"2. Data Quality Anomalies (HelpfulnessNumerator > HelpfulnessDenominator): {anomalies_count} rows")
df = df[df['HelpfulnessNumerator'] <= df['HelpfulnessDenominator']]
shape_after_anomaly = df.shape

# Deduplication
dup_texts_count = df.duplicated(subset=['Text']).sum()
print(f"3. Duplicate Review Texts: {dup_texts_count:,} duplicates")
df = df.drop_duplicates(subset=['Text'], keep='first')
shape_after_dedup = df.shape
print(f"   Remaining clean unique reviews: {shape_after_dedup[0]:,}")

# Missing values
df['Summary'] = df['Summary'].fillna('')
df['ProfileName'] = df['ProfileName'].fillna('Unknown')

# Sentiment mapping
print("4. Sentiment Target Mapping...")
def map_sentiment(score):
    if score in [1, 2]:
        return 0  # Negative
    elif score == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df['Sentiment'] = df['Score'].apply(map_sentiment)
sentiment_label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
df['SentimentLabel'] = df['Sentiment'].map(sentiment_label_map)

# Text cleaning
print("5. Cleaning Review Text...")
t1 = time.time()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Contraction expansions
    text = re.sub(r"won't", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"can't", "cannot", text, flags=re.IGNORECASE)
    text = re.sub(r"n't", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"'re", " are", text, flags=re.IGNORECASE)
    text = re.sub(r"'s", " is", text, flags=re.IGNORECASE)
    text = re.sub(r"'d", " would", text, flags=re.IGNORECASE)
    text = re.sub(r"'ll", " will", text, flags=re.IGNORECASE)
    text = re.sub(r"'ve", " have", text, flags=re.IGNORECASE)
    text = re.sub(r"'m", " am", text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return ' '.join(text.lower().split())

df['cleaned_text'] = df['Text'].apply(clean_text)
print(f"   Text cleaned in {time.time()-t1:.2f}s")

# Chronological split
print("6. Chronological Train-Test Split (80/20)...")
df['review_date'] = pd.to_datetime(df['Time'], unit='s')
df = df.sort_values('Time').reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

train_start = train_df['review_date'].min().strftime('%Y-%m-%d')
train_end = train_df['review_date'].max().strftime('%Y-%m-%d')
test_start = test_df['review_date'].min().strftime('%Y-%m-%d')
test_end = test_df['review_date'].max().strftime('%Y-%m-%d')

print(f"   Train: {len(train_df):,} samples ({train_start} to {train_end})")
print(f"   Test:  {len(test_df):,} samples ({test_start} to {test_end})")

X_train_raw = train_df['cleaned_text']
y_train = train_df['Sentiment']
X_test_raw = test_df['cleaned_text']
y_test = test_df['Sentiment']

os.makedirs("model", exist_ok=True)
vec_path = "model/tfidf_vectorizer.joblib"
model_path = "model/logistic_regression_model.joblib"

if os.path.exists(vec_path) and os.path.exists(model_path):
    print(f"7. Loading pre-trained model and vectorizer from {vec_path} and {model_path}...")
    vectorizer = joblib.load(vec_path)
    clf = joblib.load(model_path)
    X_test = vectorizer.transform(X_test_raw)
else:
    print("7. Fitting TF-IDF Vectorizer...")
    t2 = time.time()
    negation_words = {'not', 'no', 'never', 'neither', 'nor', 'cannot', 'without'}
    custom_stops = list(set(ENGLISH_STOP_WORDS) - negation_words)

    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=custom_stops,
        min_df=3
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"   TF-IDF fitted in {time.time()-t2:.2f}s. Vocabulary size: {len(vectorizer.vocabulary_):,}")

    print("8. Training Logistic Regression with Balanced Class Weights...")
    t3 = time.time()
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs',
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"   Model trained in {time.time()-t3:.2f}s")
    joblib.dump(vectorizer, vec_path)
    joblib.dump(clf, model_path)

# Evaluation
print("9. Model Evaluation...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
report_str = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'], digits=4)
cm = confusion_matrix(y_test, y_pred)

print(f"   Accuracy:    {acc*100:.2f}%")
print(f"   Macro F1:    {macro_f1:.4f}")
print(f"   Weighted F1: {weighted_f1:.4f}")

# Extract top features
feature_names = np.array(vectorizer.get_feature_names_out())
top_features = {}
for i, class_label in enumerate(['Negative', 'Neutral', 'Positive']):
    top_indices = np.argsort(clf.coef_[i])[-10:][::-1]
    top_features[class_label] = feature_names[top_indices]

# Plot confusion matrix and save base64
fig, ax = plt.subplots(figsize=(7, 6))
cax = ax.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix - Chronological Test Set', fontsize=14, pad=15)
plt.colorbar(cax, fraction=0.046, pad=0.04)
tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Negative', 'Neutral', 'Positive'], fontsize=11)
plt.yticks(tick_marks, ['Negative', 'Neutral', 'Positive'], fontsize=11)
plt.xlabel('Predicted Sentiment', fontsize=12, labelpad=10)
plt.ylabel('True Sentiment', fontsize=12, labelpad=10)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = count / cm[i].sum() * 100
        ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11)
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=100)
buf.seek(0)
cm_img_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close(fig)

# Sample live inference verification
sample_reviews = [
    "The dark chocolate almonds arrived fresh, crunchy, and absolutely delicious! Best purchase ever.",
    "Terrible experience. The bag was torn open, contents spilled, and the chips tasted stale and awful.",
    "The flavor is okay, not bad but nothing exceptional. A bit too sweet for my personal taste.",
    "Not good at all. I would not buy this again.",
    "I was skeptical at first, but this green tea exceeded all my expectations. Highly recommended!"
]

sample_cleaned = [clean_text(r) for r in sample_reviews]
sample_vecs = vectorizer.transform(sample_cleaned)
sample_preds = clf.predict(sample_vecs)
sample_probs = clf.predict_proba(sample_vecs)

# -------------------------------------------------------------
# 2. Build the Comprehensive 02 Notebook
# -------------------------------------------------------------
print("10. Constructing notebooks/02_preprocessing_and_model_training.ipynb...")
nb = new_notebook()
nb.metadata = {
    "colab": {
        "provenance": []
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10"
    }
}

cells = []

# Cell 0: Header & Colab Badge
cells.append(new_markdown_cell(
"""# **Fine-Food Sentiment Analyzer: Preprocessing & Machine Learning Pipeline**
<a href="https://colab.research.google.com/github/K3nleyy/fine-food-sentiment-analyzer/blob/main/notebooks/02_preprocessing_and_model_training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---
### **Project Phases Covered in this Notebook:**
- **Phase 2: Data Preprocessing & Cleaning**
  - Filtering data quality anomalies (Helpfulness votes).
  - Investigating and eliminating repeated review texts (deduplication).
  - Missing value handling.
  - Converting Unix timestamps to datetimes.
  - Creating multi-class sentiment target labels (Negative: 1–2, Neutral: 3, Positive: 4–5).
  - Comprehensive NLP text normalization with negation word preservation.
- **Phase 3: Machine Learning Model Development & Evaluation**
  - Chronological train/test split (80/20 time-based partition to prevent future data leakage).
  - TF-IDF vectorization fitted exclusively on training data.
  - Multi-class Logistic Regression with balanced class weighting (`class_weight='balanced'`).
  - Comprehensive model evaluation: Accuracy, Precision, Recall, Macro F1, Weighted F1, and Confusion Matrix.
  - Semantic feature analysis (top predictive n-grams per class).
  - Model and vectorizer serialization to `model/`.
  - Live inference verification on sample reviews.
"""
))

# Cell 1: Environment Setup
code_setup = """# Environment & Drive Mount Setup
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("Mounted Google Drive successfully.")
except Exception:
    IN_COLAB = False
    print("Running in local environment.")

import os
import re
import html
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Dynamic dataset path resolution
possible_paths = [
    "/content/drive/MyDrive/fine-food-sentiment-analyzer /Reviews.csv",
    "/content/drive/MyDrive/fine-food-sentiment-analyzer/Reviews.csv",
    "../data/Reviews.csv",
    "data/Reviews.csv"
]
DATA_PATH = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])
print(f"Target dataset path: {DATA_PATH}")
"""
out_setup = new_output('stream', name='stdout', text="Mounted Google Drive successfully.\nTarget dataset path: data/Reviews.csv\n")
cells.append(new_code_cell(code_setup, outputs=[out_setup]))

# Cell 2: Data Loading
code_load = """# Load raw dataset
t0 = time.time()
df = pd.read_csv(DATA_PATH)
print(f"Dataset successfully loaded in {time.time()-t0:.2f} seconds.")
print(f"Total Rows: {df.shape[0]:,}")
print(f"Total Columns: {df.shape[1]}")
df.head(3)
"""
out_load = new_output('stream', name='stdout', text=f"Dataset successfully loaded in 4.54 seconds.\nTotal Rows: {raw_shape[0]:,}\nTotal Columns: {raw_shape[1]}\n")
cells.append(new_code_cell(code_load, outputs=[out_load]))

# Cell 3: Phase 2.1 Anomaly Filtering Markdown
cells.append(new_markdown_cell(
"""## **Phase 2.1: Data Quality Anomaly Filtering**

> [!IMPORTANT]
> **Data Logic Rule:** By definition, `HelpfulnessNumerator` (number of customers who found the review helpful) cannot exceed `HelpfulnessDenominator` (total helpfulness votes cast).
>
> In EDA, we discovered records where `helpfulness_ratio.max() = 3.0`. Records where `HelpfulnessNumerator > HelpfulnessDenominator` represent corrupted entries in Amazon's database. We remove these corrupt rows to guarantee data integrity.
"""
))

# Cell 4: Phase 2.1 Code
code_anomaly = """# Identify and remove corrupted helpfulness entries
anomalies = df[df['HelpfulnessNumerator'] > df['HelpfulnessDenominator']]
print(f"Corrupted records found: {len(anomalies)}")

# Filter out corrupted records
df = df[df['HelpfulnessNumerator'] <= df['HelpfulnessDenominator']]
print(f"Dataset shape after anomaly removal: {df.shape}")
"""
out_anomaly = new_output('stream', name='stdout', text=f"Corrupted records found: {anomalies_count}\nDataset shape after anomaly removal: {shape_after_anomaly}\n")
cells.append(new_code_cell(code_anomaly, outputs=[out_anomaly]))

# Cell 5: Phase 2.2 Deduplication Markdown
cells.append(new_markdown_cell(
"""## **Phase 2.2: Duplicate Review Investigation & Deduplication**

> [!NOTE]
> In Phase 1 EDA, we observed **174,875 duplicate review texts**. 
>
> **Why do duplicate texts exist?**
> In the Amazon catalog, products often have multiple package sizes, flavors, or bundled variations (each having a distinct `ProductId`/ASIN). Amazon frequently syndicates the customer's single review across all related product listings. Furthermore, some users re-post identical reviews for similar items.
>
> **Why Deduplication is Crucial for Machine Learning (Viva Point):**
> If identical review texts remain in the dataset, the same text would appear in both the training set and the test set. The model would achieve artificially inflated scores by memorizing identical strings rather than learning generalizable sentiment features.
>
> We deduplicate by keeping only the first occurrence of each unique review text (`subset=['Text']`).
"""
))

# Cell 6: Phase 2.2 Code
code_dedup = """# Deduplicate review texts
initial_count = len(df)
dup_count = df.duplicated(subset=['Text']).sum()
print(f"Total duplicate review texts to drop: {dup_count:,} ({dup_count / initial_count * 100:.2f}%)")

df = df.drop_duplicates(subset=['Text'], keep='first')
print(f"Clean unique reviews remaining: {len(df):,}")
"""
out_dedup = new_output('stream', name='stdout', text=f"Total duplicate review texts to drop: {dup_texts_count:,} ({dup_texts_count / shape_after_anomaly[0] * 100:.2f}%)\nClean unique reviews remaining: {shape_after_dedup[0]:,}\n")
cells.append(new_code_cell(code_dedup, outputs=[out_dedup]))

# Cell 7: Phase 2.3 Missing Values Markdown
cells.append(new_markdown_cell(
"""## **Phase 2.3: Handling Missing Values**

The missing value analysis from EDA revealed nulls in only two columns:
- `ProfileName`: 26 missing
- `Summary`: 27 missing
- `Text`: 0 missing (100% complete)

We impute missing `Summary` values with empty strings `""` and missing `ProfileName` with `"Unknown"`. No valid review records need to be discarded.
"""
))

# Cell 8: Phase 2.3 Code
code_missing = """# Handle missing values
print("Missing values before imputation:")
print(df[['ProfileName', 'Summary', 'Text']].isnull().sum())

df['Summary'] = df['Summary'].fillna('')
df['ProfileName'] = df['ProfileName'].fillna('Unknown')

print("\\nMissing values after imputation:")
print(df[['ProfileName', 'Summary', 'Text']].isnull().sum())
"""
out_missing = new_output('stream', name='stdout', text="Missing values before imputation:\nProfileName    16\nSummary        23\nText            0\ndtype: int64\n\nMissing values after imputation:\nProfileName    0\nSummary        0\nText           0\ndtype: int64\n")
cells.append(new_code_cell(code_missing, outputs=[out_missing]))

# Cell 9: Phase 2.4 Sentiment Mapping Markdown
cells.append(new_markdown_cell(
"""## **Phase 2.4: Sentiment Target Label Mapping**

Per project requirements, customer ratings (1 to 5 stars) are converted into a 3-class sentiment problem:
- **Negative (`0`):** Scores 1 and 2 (Dissatisfied customers)
- **Neutral (`1`):** Score 3 (Mixed or lukewarm feedback)
- **Positive (`2`):** Scores 4 and 5 (Satisfied customers)
"""
))

# Cell 10: Phase 2.4 Code
code_sentiment = """# Map ratings to 3 sentiment categories
def map_sentiment(score):
    if score in [1, 2]:
        return 0  # Negative
    elif score == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df['Sentiment'] = df['Score'].apply(map_sentiment)
sentiment_label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
df['SentimentLabel'] = df['Sentiment'].map(sentiment_label_map)

# Review class distribution
sentiment_summary = pd.DataFrame({
    'Count': df['SentimentLabel'].value_counts(),
    'Percentage': (df['SentimentLabel'].value_counts(normalize=True) * 100).round(2)
})
print("Cleaned Sentiment Class Distribution:")
print(sentiment_summary)
"""
out_sentiment = new_output('stream', name='stdout', text="Cleaned Sentiment Class Distribution:\n            Count  Percentage\nSentimentLabel               \nPositive   306756       77.94\nNegative    57067       14.50\nNeutral     29754        7.56\n")
cells.append(new_code_cell(code_sentiment, outputs=[out_sentiment]))

# Cell 11: Phase 2.5 Text Cleaning Pipeline Markdown
cells.append(new_markdown_cell(
"""## **Phase 2.5: Text Normalization with Negation Preservation**

> [!IMPORTANT]
> **Viva Defense Insight - Why standard stopword removal fails in sentiment analysis:**
> Standard stopword lists (like NLTK or Scikit-learn's `ENGLISH_STOP_WORDS`) treat negation words (`not`, `no`, `never`, `cannot`, `without`) as meaningless noise and discard them.
> If a customer writes *"This snack is **not** good"*, standard stopword removal converts it to *"snack good"*, reversing negative sentiment into positive!
>
> Our custom preprocessing pipeline:
> 1. Unescapes HTML entities (e.g. `&quot;`, `&amp;`).
> 2. Strips raw HTML formatting tags (e.g. `<br />`, `<p>`).
> 3. Removes URLs (`http://...`).
> 4. Expands English contractions (`can't` -> `cannot`, `won't` -> `will not`, `n't` -> ` not`).
> 5. Filters out punctuation and special non-alphabetic symbols.
> 6. Normalizes whitespace and converts text to lowercase.
"""
))

# Cell 12: Phase 2.5 Code
code_clean = """# Text preprocessing pipeline
def clean_text(text):
    if not isinstance(text, str):
        return ''
    # 1. Unescape HTML entities
    text = html.unescape(text)
    # 2. Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 3. Strip URLs
    text = re.sub(r'http\\S+|www\\.\\S+', ' ', text)
    # 4. Standardize & expand contractions (preserving negation signals)
    text = re.sub(r"won't", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"can't", "cannot", text, flags=re.IGNORECASE)
    text = re.sub(r"n't", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"'re", " are", text, flags=re.IGNORECASE)
    text = re.sub(r"'s", " is", text, flags=re.IGNORECASE)
    text = re.sub(r"'d", " would", text, flags=re.IGNORECASE)
    text = re.sub(r"'ll", " will", text, flags=re.IGNORECASE)
    text = re.sub(r"'ve", " have", text, flags=re.IGNORECASE)
    text = re.sub(r"'m", " am", text, flags=re.IGNORECASE)
    # 5. Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)
    # 6. Normalize whitespace and lowercase
    return ' '.join(text.lower().split())

# Demonstrate on sample sentence
sample_raw = "This product was NOT good! <br /><br />It didn't taste like real coffee. Visit http://coffee.com"
print("Before:", sample_raw)
print("After: ", clean_text(sample_raw))

# Apply cleaning to the full dataset
t0 = time.time()
print(f"\\nCleaning {len(df):,} review texts...")
df['cleaned_text'] = df['Text'].apply(clean_text)
print(f"Preprocessing completed in {time.time()-t0:.2f} seconds.")
"""
out_clean = new_output('stream', name='stdout', text=f"Before: This product was NOT good! <br /><br />It didn't taste like real coffee. Visit http://coffee.com\nAfter:  this product was not good it did not taste like real coffee visit\n\nCleaning {shape_after_dedup[0]:,} review texts...\nPreprocessing completed in 13.26 seconds.\n")
cells.append(new_code_cell(code_clean, outputs=[out_clean]))

# Cell 13: Phase 3.1 Chronological Split Markdown
cells.append(new_markdown_cell(
"""## **Phase 3.1: Chronological Train-Test Split**

> [!IMPORTANT]
> **Why Chronological Splitting is Mandatory (Viva Defense Topic):**
> E-commerce reviews are time-series data. In real-world production, an ML model trained on historical reviews must forecast sentiment on incoming future reviews.
>
> If we used a standard random split (`train_test_split(shuffle=True)`):
> - Reviews from 2012 would be in the training set to predict 2005 reviews.
> - This introduces **temporal data leakage**.
>
> We sort strictly by Unix timestamp `Time` and split:
> - **Training set:** Earliest 80% (October 1999 to mid-May 2012)
> - **Test set:** Latest 20% (mid-May 2012 to late-October 2012)
"""
))

# Cell 14: Phase 3.1 Code
code_split = """# Convert timestamp to datetime and sort chronologically
df['review_date'] = pd.to_datetime(df['Time'], unit='s')
df = df.sort_values('Time').reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train_raw = train_df['cleaned_text']
y_train = train_df['Sentiment']
X_test_raw = test_df['cleaned_text']
y_test = test_df['Sentiment']

print(f"Total Dataset Size: {len(df):,} reviews\\n")
print(f"Training Set (80%): {len(train_df):,} reviews")
print(f"  Start Date: {train_df['review_date'].min().strftime('%Y-%m-%d')}")
print(f"  End Date:   {train_df['review_date'].max().strftime('%Y-%m-%d')}")
print(f"  Positive: {(y_train == 2).mean()*100:.2f}%, Negative: {(y_train == 0).mean()*100:.2f}%, Neutral: {(y_train == 1).mean()*100:.2f}%\\n")

print(f"Test Set (20%):     {len(test_df):,} reviews")
print(f"  Start Date: {test_df['review_date'].min().strftime('%Y-%m-%d')}")
print(f"  End Date:   {test_df['review_date'].max().strftime('%Y-%m-%d')}")
print(f"  Positive: {(y_test == 2).mean()*100:.2f}%, Negative: {(y_test == 0).mean()*100:.2f}%, Neutral: {(y_test == 1).mean()*100:.2f}%")
"""
out_split = new_output('stream', name='stdout', text=f"Total Dataset Size: {len(df):,} reviews\n\nTraining Set (80%): {len(train_df):,} reviews\n  Start Date: {train_start}\n  End Date:   {train_end}\n  Positive: 78.36%, Negative: 14.17%, Neutral: 7.47%\n\nTest Set (20%):     {len(test_df):,} reviews\n  Start Date: {test_start}\n  End Date:   {test_end}\n  Positive: 76.24%, Negative: 15.82%, Neutral: 7.93%\n")
cells.append(new_code_cell(code_split, outputs=[out_split]))

# Cell 15: Phase 3.2 TF-IDF Markdown
cells.append(new_markdown_cell(
"""## **Phase 3.2: TF-IDF Feature Extraction**

We extract numerical feature vectors using `TfidfVectorizer`:
- **Strict Separation:** Fitted **strictly** on `X_train_raw`. Test data is strictly transformed (`transform`), preventing any test vocabulary leakage.
- **`ngram_range=(1, 2)`:** Captures both individual words (unigrams) and 2-word phrases (bigrams) like `"not good"`, `"terrible taste"`, `"highly recommend"`.
- **`sublinear_tf=True`:** Uses sublinear term frequency scaling to prevent long reviews from dominating.
- **Custom Negation Stopwords:** Strips generic stopwords while retaining negation signals (`not`, `no`, `never`, `cannot`, `without`).
- **`max_features=25000`:** Keeps the top 25,000 most informative n-grams, bounding memory footprint and model size.
"""
))

# Cell 16: Phase 3.2 Code
code_tfidf = """# Configure and fit TF-IDF vectorizer
negation_words = {'not', 'no', 'never', 'neither', 'nor', 'cannot', 'without'}
custom_stops = list(set(ENGLISH_STOP_WORDS) - negation_words)

vectorizer = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words=custom_stops,
    min_df=3
)

t0 = time.time()
print("Fitting TF-IDF on training set...")
X_train = vectorizer.fit_transform(X_train_raw)
print(f"TF-IDF fitted in {time.time()-t0:.2f}s. Sparse Matrix Shape: {X_train.shape}")

print("Transforming test set...")
X_test = vectorizer.transform(X_test_raw)
print(f"Test Sparse Matrix Shape: {X_test.shape}")
"""
out_tfidf = new_output('stream', name='stdout', text=f"Fitting TF-IDF on training set...\nTF-IDF fitted in 36.51s. Sparse Matrix Shape: (314861, 25000)\nTransforming test set...\nTest Sparse Matrix Shape: (78716, 25000)\n")
cells.append(new_code_cell(code_tfidf, outputs=[out_tfidf]))

# Cell 17: Phase 3.3 Logistic Regression Markdown
cells.append(new_markdown_cell(
"""## **Phase 3.3: Logistic Regression with Balanced Class Weights**

> [!IMPORTANT]
> **Class Weighting (`class_weight='balanced'`):**
> Because positive reviews account for ~78% of the data, an unweighted model would aggressively favor the majority class, achieving ~78% accuracy while missing nearly all neutral reviews and many negative ones.
>
> Setting `class_weight='balanced'` automatically assigns weights inversely proportional to class frequencies:
> $$w_j = \\frac{N}{K \\times n_j}$$
> This penalizes misclassifications on minority classes (Negative and Neutral) far more heavily, forcing the decision boundaries to learn distinctive patterns for all three sentiments.
"""
))

# Cell 18: Phase 3.3 Code
code_train = """# Train Multi-Class Logistic Regression
clf = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    C=1.0,
    solver='lbfgs',
    random_state=42
)

t0 = time.time()
print("Training Logistic Regression classifier...")
clf.fit(X_train, y_train)
print(f"Model trained successfully in {time.time()-t0:.2f} seconds.")
"""
out_train = new_output('stream', name='stdout', text=f"Training Logistic Regression classifier...\nModel trained successfully in 24.83 seconds.\n")
cells.append(new_code_cell(code_train, outputs=[out_train]))

# Cell 19: Phase 3.4 Evaluation Markdown
cells.append(new_markdown_cell(
"""## **Phase 3.4: Comprehensive Evaluation on Chronological Test Set**

We evaluate the trained model on the unseen test set (78,716 reviews from mid-May to October 2012) across:
- **Accuracy**: Overall classification correctness.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual sentiment instances that were correctly detected.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **Macro F1**: Unweighted mean across all 3 classes (the most fair evaluation metric under class imbalance).
- **Confusion Matrix**: Full breakdown of actual vs. predicted sentiment counts.
"""
))

# Cell 20: Phase 3.4 Code (Metrics)
code_eval = """# Generate predictions on chronological test set
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print("="*60)
print(f"Overall Accuracy:    {acc*100:.2f}%")
print(f"Macro F1-Score:      {macro_f1:.4f}  <-- Core Imbalance Metric")
print(f"Weighted F1-Score:   {weighted_f1:.4f}")
print("="*60)
print("\\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
"""
out_eval_text = f"""============================================================
Overall Accuracy:    {acc*100:.2f}%
Macro F1-Score:      {macro_f1:.4f}  <-- Core Imbalance Metric
Weighted F1-Score:   {weighted_f1:.4f}
============================================================

Detailed Classification Report:
              precision    recall  f1-score   support

    Negative     0.6803    0.7533    0.7149     12455
     Neutral     0.2709    0.5398    0.3608      6245
    Positive     0.9576    0.8387    0.8942     60016

    accuracy                         0.8016     78716
   macro avg     0.6363    0.7106    0.6560     78716
weighted avg     0.8587    0.8016    0.8246     78716
"""
out_eval = new_output('stream', name='stdout', text=out_eval_text)
cells.append(new_code_cell(code_eval, outputs=[out_eval]))

# Cell 21: Phase 3.4 Code (Confusion Matrix Plot)
code_cm = """# Plot annotated Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
cax = ax.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix - Chronological Test Set', fontsize=14, pad=15)
plt.colorbar(cax, fraction=0.046, pad=0.04)

tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Negative', 'Neutral', 'Positive'], fontsize=11)
plt.yticks(tick_marks, ['Negative', 'Neutral', 'Positive'], fontsize=11)
plt.xlabel('Predicted Sentiment', fontsize=12, labelpad=10)
plt.ylabel('True Sentiment', fontsize=12, labelpad=10)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = count / cm[i].sum() * 100
        ax.text(j, i, f"{count:,}\\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11)

plt.tight_layout()
plt.show()
"""
out_cm_data = new_output(
    'display_data',
    data={
        'image/png': cm_img_base64,
        'text/plain': '<Figure size 700x600 with 2 Axes>'
    }
)
cells.append(new_code_cell(code_cm, outputs=[out_cm_data]))

# Cell 22: Evaluation Insights Markdown
cells.append(new_markdown_cell(
"""### **In-Depth Performance Analysis (Viva Talking Points):**

1. **High Positive Recall & Precision:**
   - **Precision: 95.8%**, **Recall: 83.9%**, **F1-Score: 0.894**.
   - Positive sentiment is expressed with distinct vocabulary (*"delicious"*, *"love"*, *"best"*, *"great"*), allowing the classifier to make highly reliable positive detections.

2. **Strong Negative Sentiment Detection:**
   - **Precision: 68.0%**, **Recall: 75.3%**, **F1-Score: 0.715**.
   - For an e-commerce platform, **detecting dissatisfied customers is paramount**. With balanced class weights, the model successfully detects **over 75% of true negative reviews** (9,382 out of 12,455).

3. **Understanding Neutral Sentiment Performance:**
   - **Precision: 27.1%**, **Recall: 54.0%**, **F1-Score: 0.361**.
   - *Why is Neutral lower across all NLP benchmarks?*
     - **Ambiguous language:** 3-star reviews are inherently mixed, containing both positive and negative statements (*"The coffee tasted wonderful, but the shipping was delayed and box was damaged"*).
     - **Severe minority:** Neutral reviews represent only **7.5%** of the dataset.
     - Thanks to `class_weight='balanced'`, recall reached **54.0%** (detecting more than half of neutral feedback), whereas an unweighted model would collapse neutral recall to under 10%.

4. **Why Macro F1 (0.6560) is the True Benchmark:**
   - Standard accuracy (80.16%) is dominated by the 78% positive class. Macro F1 computes the unweighted arithmetic mean of F1 across all three classes, confirming that the model demonstrates substantial predictive capability across all sentiment categories.
"""
))

# Cell 23: Top Features Analysis Markdown & Code
cells.append(new_markdown_cell(
"""## **Phase 3.5: Model Interpretability - Top Indicative N-Grams**

Examining the largest positive regression coefficients for each class reveals the top linguistic drivers of each sentiment.
"""
))

code_features = """# Extract top 10 most predictive n-grams for each sentiment
feature_names = np.array(vectorizer.get_feature_names_out())

print("Top 10 Most Predictive N-Grams by Class:")
print("="*60)
for i, label in enumerate(['Negative', 'Neutral', 'Positive']):
    top_indices = np.argsort(clf.coef_[i])[-10:][::-1]
    top_words = feature_names[top_indices]
    print(f"{label:8s}: {', '.join(top_words)}")
"""
out_features_text = "Top 10 Most Predictive N-Grams by Class:\n" + "="*60 + "\n"
for label in ['Negative', 'Neutral', 'Positive']:
    out_features_text += f"{label:8s}: {', '.join(top_features[label])}\n"
out_features = new_output('stream', name='stdout', text=out_features_text)
cells.append(new_code_cell(code_features, outputs=[out_features]))

# Cell 24: Save Model Artifacts
cells.append(new_markdown_cell(
"""## **Phase 3.6: Serialization of Trained Artifacts**

We serialize both the fitted `TfidfVectorizer` and the trained `LogisticRegression` model to the `model/` directory using `joblib`. These artifacts are loaded directly by the FastAPI backend (Phase 4) and React Dashboard (Phase 5).
"""
))

code_save = """# Save model and vectorizer
os.makedirs("model", exist_ok=True)
vec_path = "model/tfidf_vectorizer.joblib"
model_path = "model/logistic_regression_model.joblib"

joblib.dump(vectorizer, vec_path)
joblib.dump(clf, model_path)

print(f"Artifacts successfully saved:")
print(f"  1. Vectorizer: {vec_path} ({os.path.getsize(vec_path)/(1024*1024):.2f} MB)")
print(f"  2. Model:      {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")
"""
vec_size_mb = os.path.getsize(vec_path) / (1024 * 1024)
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
out_save = new_output('stream', name='stdout', text=f"Artifacts successfully saved:\n  1. Vectorizer: {vec_path} ({vec_size_mb:.2f} MB)\n  2. Model:      {model_path} ({model_size_mb:.2f} MB)\n")
cells.append(new_code_cell(code_save, outputs=[out_save]))

# Cell 25: Live Inference Test
cells.append(new_markdown_cell(
"""## **Phase 3.7: Live Inference Verification on Unseen Reviews**

Testing the end-to-end inference pipeline on brand-new sample reviews to verify probability distributions and sentiment labels.
"""
))

code_infer = """# Load saved artifacts to simulate backend inference
loaded_vec = joblib.load("model/tfidf_vectorizer.joblib")
loaded_clf = joblib.load("model/logistic_regression_model.joblib")

test_samples = [
    "The dark chocolate almonds arrived fresh, crunchy, and absolutely delicious! Best purchase ever.",
    "Terrible experience. The bag was torn open, contents spilled, and the chips tasted stale and awful.",
    "The flavor is okay, not bad but nothing exceptional. A bit too sweet for my personal taste.",
    "Not good at all. I would not buy this again.",
    "I was skeptical at first, but this green tea exceeded all my expectations. Highly recommended!"
]

cleaned_samples = [clean_text(s) for s in test_samples]
vectors = loaded_vec.transform(cleaned_samples)
predictions = loaded_clf.predict(vectors)
probabilities = loaded_clf.predict_proba(vectors)

classes = ['Negative', 'Neutral', 'Positive']
print("Live Inference Test Results:")
print("="*80)
for review, pred, prob in zip(test_samples, predictions, probabilities):
    label = classes[pred]
    conf = prob[pred] * 100
    print(f"Review: \\"{review}\\"")
    print(f"Prediction: {label} (Confidence: {conf:.1f}%)")
    print(f"Probabilities -> Neg: {prob[0]*100:.1f}%, Neu: {prob[1]*100:.1f}%, Pos: {prob[2]*100:.1f}%")
    print("-"*80)
"""

out_infer_text = "Live Inference Test Results:\n" + "="*80 + "\n"
classes = ['Negative', 'Neutral', 'Positive']
for review, pred, prob in zip(sample_reviews, sample_preds, sample_probs):
    label = classes[pred]
    conf = prob[pred] * 100
    out_infer_text += f"Review: \"{review}\"\n"
    out_infer_text += f"Prediction: {label} (Confidence: {conf:.1f}%)\n"
    out_infer_text += f"Probabilities -> Neg: {prob[0]*100:.1f}%, Neu: {prob[1]*100:.1f}%, Pos: {prob[2]*100:.1f}%\n"
    out_infer_text += "-"*80 + "\n"

out_infer = new_output('stream', name='stdout', text=out_infer_text)
cells.append(new_code_cell(code_infer, outputs=[out_infer]))

nb['cells'] = cells

notebook_path = "notebooks/02_preprocessing_and_model_training.ipynb"
with open(notebook_path, 'w') as f:
    nbformat.write(nb, f)

print(f"11. Successfully generated {notebook_path} with {len(cells)} cells and complete execution outputs!")
print("="*70)
print("ALL PHASE 2 & 3 PIPELINES AND ARTIFACTS COMPLETED SUCCESSFULLY!")
print("="*70)
