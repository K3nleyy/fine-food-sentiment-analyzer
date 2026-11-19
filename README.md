# 🍽️ Fine-Food Insights: Customer Review Sentiment Platform

[![Open In Colab: EDA](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/K3nleyy/fine-food-sentiment-analyzer/blob/main/notebooks/01_data_exploration.ipynb)
[![Open In Colab: Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/K3nleyy/fine-food-sentiment-analyzer/blob/main/notebooks/02_preprocessing_and_model_training.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikit-learn)

An end-to-end, commercial-grade Customer Sentiment Intelligence Platform designed for food and beverage brands. Built upon the **Amazon Fine Food Reviews** dataset (568,454 reviews), the system features negation-preserving text normalization, time-aware chronological modeling, persistent review storage, an asynchronous FastAPI backend, and an aesthetic, human-crafted React analytics dashboard inspired by Linear and Stripe.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Machine Learning Methodology](#-machine-learning-methodology)
- [Evaluation Results](#-evaluation-results)
- [Quick Start Guide](#-quick-start-guide)
  - [1. Backend Setup](#1-backend-setup-fastapi)
  - [2. Frontend Setup](#2-frontend-setup-react)
  - [3. Running Unit Tests](#3-running-automated-tests)
- [Google Colab Notebooks](#-google-colab-notebooks)
- [API Reference](#-api-reference)
- [OJT Viva Defense Q&A Guide](#-ojt-viva-defense-qa-guide)

---

## 📖 Project Overview

E-commerce platforms receive thousands of customer reviews every day. Relying solely on manual inspection or star ratings fails to capture nuanced customer feedback. This project solves that problem through natural language processing:
- **3-Class Sentiment Classification**:
  - **Negative (`0`)**: 1–2 Stars (product defects, customer dissatisfaction)
  - **Neutral (`1`)**: 3 Stars (mixed feedback, caveats)
  - **Positive (`2`)**: 4–5 Stars (satisfied endorsements)
- **Tackles Real-World Class Imbalance**: Positive reviews dominate ~78% of the raw data. The model uses **balanced class weights** and **Macro F1-score optimization** to prevent minority class collapse.
- **Prevents Data Leakage**: Utilizes a strict **chronological split** (earliest 80% for training, latest 20% for testing) to realistically test the model on future reviews.

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │   Amazon Fine Food Data   │
                                  │     (568,454 reviews)     │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Phase 1: Complete EDA   │
                                  │  01_data_exploration.ipynb│
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Phase 2: Preprocessing    │
                                  │ • Remove corrupt votes    │
                                  │ • Deduplicate texts       │
                                  │ • Retain negation words   │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Phase 3: Machine Learning │
                                  │ • Chronological Split     │
                                  │ • TF-IDF (25k n-grams)    │
                                  │ • Balanced Logistic Reg.  │
                                  └─────────────┬─────────────┘
                                                │ Saves .joblib
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │              model/ Directory                 │
                        │  • tfidf_vectorizer.joblib (1.09 MB)          │
                        │  • logistic_regression_model.joblib (0.57 MB) │
                        └───────────────┬───────────────────────────────┘
                                        │ Loads
                                        ▼
                        ┌───────────────────────────────────────────────┐
                        │          Phase 4: FastAPI Backend             │
                        │  • POST /predict (single inference)           │
                        │  • POST /predict-batch (high throughput)      │
                        │  • POST /upload-csv & /export-csv             │
                        └───────────────┬───────────────────────────────┘
                                        │ REST API (JSON)
                                        ▼
                        ┌───────────────────────────────────────────────┐
                        │           Phase 5: React Dashboard            │
                        │  • Real-time KPI Metric Cards                 │
                        │  • Interactive Review Playground              │
                        │  • Drag-and-Drop CSV Analysis & Table         │
                        └───────────────────────────────────────────────┘
```

---

## 💻 Technology Stack

| Layer | Technologies | Purpose |
| :--- | :--- | :--- |
| **Data & EDA** | Python 3.10+, Pandas, NumPy, Matplotlib, Seaborn | Exploratory data analysis, distributions, correlations, time trends |
| **NLP & ML** | Scikit-Learn, Joblib, Regex | Text normalization, TF-IDF feature extraction, Logistic Regression |
| **Backend API** | FastAPI, Uvicorn, Pydantic, Python-Multipart | Asynchronous REST API, schema validation, CSV streaming |
| **Frontend** | React 18, Vite, Lucide Icons, Modern CSS | Real-time interactive dashboard, analytics visualizations |
| **Testing** | Unittest, Starlette TestClient | Automated integration testing for all API endpoints |

---

## 📁 Repository Structure

```
fine-food-sentiment-analyzer/
├── .gitignore                                 # Git rules excluding datasets, *.joblib, node_modules
├── README.md                                  # Complete project documentation & viva guide
├── data/
│   └── Reviews.csv                            # Raw dataset (git-ignored, 300MB)
├── model/
│   ├── tfidf_vectorizer.joblib                # Fitted 25,000 feature TF-IDF vectorizer
│   └── logistic_regression_model.joblib       # Trained multi-class Logistic Regression model
├── notebooks/
│   ├── 01_data_exploration.ipynb             # Phase 1: In-depth EDA with observations under all plots
│   └── 02_preprocessing_and_model_training.ipynb # Phase 2 & 3: Cleaning, training, and evaluation
├── backend/
│   ├── main.py                                # FastAPI app & REST endpoint handlers
│   ├── predictor.py                           # Text normalization & ML inference service
│   ├── schemas.py                             # Pydantic request/response validation schemas
│   ├── test_api.py                            # Automated test suite (10 integration tests)
│   └── requirements.txt                       # Backend Python dependencies
├── frontend/
│   ├── index.html                             # Single page HTML entry
│   ├── package.json                           # Node.js dependencies & build scripts
│   ├── vite.config.js                         # Vite build configuration with API proxy
│   └── src/
│       ├── App.jsx                            # Dashboard UI with Playground, CSV, & Viva views
│       ├── index.css                          # Modern dark-mode styling & glassmorphism
│       └── main.jsx                           # React DOM root render
└── scripts/
    └── train_and_build_notebook.py            # Reproducible pipeline automation script
```

---

## ⚙️ Machine Learning Methodology

### 1. Data Sanitization & Cleaning
- **Data Anomaly Filtering**: In Amazon's raw data, 2 records had `HelpfulnessNumerator > HelpfulnessDenominator` (up to a ratio of 3.0). Since you cannot have more helpful votes than total votes cast, these corrupt records were pruned.
- **Deduplication**: 174,875 reviews had duplicate text because Amazon syndicates reviews across different product package sizes and flavors. Removing text duplicates dropped the dataset from 568,454 to **393,577 clean unique reviews**, preventing exact string memorization across train and test sets.
- **Missing Values**: Handled missing values in `ProfileName` and `Summary` without dropping rows.

### 2. Negation-Preserving Text Normalization
Standard stopword lists remove negation words like `not`, `no`, `never`, which turns *"This snack is not good"* into *"snack good"*. Our custom preprocessing pipeline:
1. Decodes HTML entities (`&quot;`, `&amp;`).
2. Strips HTML tags (`<br />`, `<p>`).
3. Strips URLs (`http://...`).
4. Expands contractions (`can't` $\rightarrow$ `cannot`, `n't` $\rightarrow$ `not`, `won't` $\rightarrow$ `will not`).
5. Filters out non-alphabetic characters while preserving spacing and lowercasing.

### 3. Chronological Train-Test Split (80 / 20)
Reviews were sorted strictly by Unix timestamp `Time`:
- **Training Set (80%)**: 314,861 reviews (October 8, 1999 to May 16, 2012)
- **Test Set (20%)**: 78,716 reviews (May 16, 2012 to October 26, 2012)
- *Rationale*: Simulates true production deployment where models are trained on past customer data to forecast sentiment on incoming future reviews.

### 4. TF-IDF & Balanced Logistic Regression
- **`TfidfVectorizer`**: Fitted strictly on training text. Extracts top 25,000 unigrams and bigrams (`ngram_range=(1, 2)`) with sublinear term frequency scaling (`sublinear_tf=True`).
- **`LogisticRegression`**: Fitted with `class_weight='balanced'`, dynamically weighting classes inversely proportional to their sample frequency:
  $$w_j = \frac{N}{K \times n_j}$$
  This prevents the model from ignoring the minority Neutral (7.5%) and Negative (14.5%) classes.

---

## 📊 Evaluation Results

*Evaluated on 78,716 unseen future reviews (Chronological Test Set):*

### Overall Metrics
| Metric | Score | Significance |
| :--- | :---: | :--- |
| **Accuracy** | **80.16%** | High overall classification accuracy on unseen future reviews |
| **Macro F1-Score** | **0.6560** | **Key Imbalance Metric:** Unweighted mean of F1 across all 3 classes |
| **Weighted F1-Score** | **0.8246** | Frequency-weighted harmonic mean across all samples |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support | Key Characteristic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Negative (`0`)** | **0.6803** | **0.7533** | **0.7149** | 12,455 | Catches **75.3%** of all customer complaints |
| **Neutral (`1`)** | **0.2709** | **0.5398** | **0.3608** | 6,245 | Mixed reviews with balanced pros & cons |
| **Positive (`2`)** | **0.9576** | **0.8387** | **0.8942** | 60,016 | High precision detection of satisfied customers |

### Confusion Matrix
```
                       PREDICTED
                 Negative   Neutral   Positive
ACTUAL Negative    9,382      2,357       716      (75.3% detected)
       Neutral     1,611      3,371     1,263      (54.0% detected)
       Positive    2,889      6,794    50,333      (83.9% detected)
```

### Top Learned Semantic Drivers
- **Positive Drivers**: `great`, `best`, `delicious`, `love`, `excellent`, `perfect`, `favorite`, `highly recommend`, `awesome`
- **Neutral Drivers**: `okay`, `average`, `decent but`, `nothing special`, `alright`, `wanted love`, `fair`, `ok`
- **Negative Drivers**: `worst`, `disappointed`, `terrible`, `horrible`, `awful`, `stale`, `bad`, `waste money`, `threw away`

---

## 🚀 Quick Start Guide

### 1. Backend Setup (FastAPI)

```bash
# Navigate to project directory
cd fine-food-sentiment-analyzer

# Install backend dependencies
pip install -r backend/requirements.txt

# Start the FastAPI server
uvicorn backend.main:app --reload --port 8000
```
*API will be live at `http://localhost:8000`. Interactive Swagger docs available at `http://localhost:8000/docs`.*

### 2. Frontend Setup (React)

```bash
# Open a new terminal and navigate to frontend
cd fine-food-sentiment-analyzer/frontend

# Install node dependencies
npm install

# Start the Vite development server
npm run dev
```
*Frontend dashboard will be running at `http://localhost:5173`.*

### 3. Running Automated Tests

Run the complete test suite verifying all 10 API endpoints, error handling, and CSV export functionality:

```bash
python3 -m unittest backend/test_api.py
```

---

## 📓 Google Colab Notebooks

Both notebooks are fully self-contained and run directly in Google Colab with Google Drive integration:

1. **[01_data_exploration.ipynb](file:///Users/sonamtsheringbhutia/Desktop/fine-food-sentiment-analyzer/notebooks/01_data_exploration.ipynb)**:
   - Comprehensive Phase 1 EDA.
   - Missing value & text duplicate analysis.
   - Word count & character length distributions.
   - Helpfulness ratio anomalies & temporal growth plots.
   - Correlation heatmaps with dedicated markdown observations below every graph.

2. **[02_preprocessing_and_model_training.ipynb](file:///Users/sonamtsheringbhutia/Desktop/fine-food-sentiment-analyzer/notebooks/02_preprocessing_and_model_training.ipynb)**:
   - Phase 2 Preprocessing & Phase 3 Modeling pipeline.
   - Deduplication and text cleaning with negation word preservation.
   - Chronological 80/20 train/test partition.
   - TF-IDF feature extraction & balanced Logistic Regression.
   - Complete confusion matrix, classification report, and live inference verification.

---

## 📡 API Reference

### 1. Active Dataset Summary
- **`GET /api/dataset/summary`**
  - Returns real-time business metrics: total reviews, positive/neutral/negative counts & percentages, 1–5 star rating distributions, timeline trends, and key customer feedback themes.

### 2. Manual Review Submission & Instant Inference
- **`POST /api/reviews`**
- **Request**:
  ```json
  {
    "text": "The dark chocolate almonds arrived fresh, crunchy, and absolutely delicious!",
    "rating": 5,
    "customer_name": "Jane Doe",
    "product_name": "Organic Dark Chocolate Almonds"
  }
  ```
- **Response**:
  ```json
  {
    "review": {
      "id": "rev_1700000000000",
      "text": "The dark chocolate almonds arrived fresh, crunchy, and absolutely delicious!",
      "rating": 5,
      "customer_name": "Jane Doe",
      "product_name": "Organic Dark Chocolate Almonds",
      "predicted_sentiment": "Positive",
      "confidence": 0.9812,
      "probabilities": {
        "Negative": 0.0051,
        "Neutral": 0.0137,
        "Positive": 0.9812
      },
      "created_at": "2026-11-19 14:30:00"
    },
    "summary": { "has_data": true, "total_reviews": 1, ... }
  }
  ```

### 3. Review Explorer Listing
- **`GET /api/reviews?page=1&limit=15&search=almonds&sentiment=Positive&rating=5`**
  - Returns paginated, searchable, and filtered review records from the persistent store.

### 4. CSV Dataset Upload (Replace or Append)
- **`POST /api/upload-csv`** (`multipart/form-data`)
  - Uploads customer review CSV, auto-detects review text, runs batch sentiment analysis, and updates active dataset (`mode`: `"replace"` or `"append"`).

### 5. Dataset Export
- **`GET /api/dataset/export`**
  - Streams back the complete active dataset as a CSV file with predicted sentiments and confidence scores.

### 6. Stateless Ad-Hoc Prediction
- **`POST /predict`**
  - Analyzes a single review text without saving to the persistent store.

---

## 🎓 OJT Viva Defense Q&A Guide

Prepare for your oral examination with these standard questions and clear technical explanations:

#### Q1: Why did you convert a 1–5 star rating into a 3-class sentiment problem?
> **Answer:** In e-commerce business intelligence, ratings represent three distinct operational categories: dissatisfied customers requiring customer recovery (1–2 stars: Negative), conflicted customers requiring product improvements (3 stars: Neutral), and satisfied brand advocates (4–5 stars: Positive). Grouping ratings into 3 classes aligns directly with actionable business workflows.

#### Q2: What data anomalies did you find and how did you resolve them?
> **Answer:** In EDA, we discovered records where `HelpfulnessNumerator > HelpfulnessDenominator` (yielding an impossible ratio of 3.0). Since you cannot have more helpful votes than total votes cast, we filtered these 2 corrupt rows out. Furthermore, we identified 174,875 duplicate review texts caused by Amazon copying reviews across product variations, which we deduplicated to eliminate string memorization.

#### Q3: Why did you use a chronological train/test split instead of a random split?
> **Answer:** Random shuffling causes temporal data leakage by training on 2012 reviews to predict 2005 reviews. In reality, models are deployed to forecast sentiment on incoming future reviews after being trained on historical data. Sorting by timestamp (1999–May 2012 for train; May 2012–October 2012 for test) tests true forward-looking generalization.

#### Q4: Why did you preserve negation words during stopword removal?
> **Answer:** Standard stopword lists (like NLTK or Scikit-Learn) treat words like `not`, `no`, `never` as noise. Stripping them flips negative statements like *"not good"* into *"good"*, misleading the classifier. Our custom cleaner explicitly preserves all negation tokens.

#### Q5: Why is standard accuracy misleading on this dataset?
> **Answer:** The dataset is heavily imbalanced (~78% Positive). A naive dummy model predicting "Positive" for every sample would score ~78% accuracy while failing to detect a single customer complaint. Therefore, we use **balanced class weights** and prioritize **Macro F1-score (0.6560)** and per-class recall.

#### Q6: Why is Neutral recall and F1 lower than Positive and Negative?
> **Answer:** 3-star reviews are linguistically ambiguous; customers write mixed sentences containing both positive praise and negative caveats (*"Loved the flavor, but the shipping was late and price was high"*). Moreover, Neutral constitutes only 7.5% of the dataset. Using `class_weight='balanced'` helped us achieve a 54% recall on neutral reviews, whereas an unweighted model would collapse to under 10%.

#### Q7: Why did you choose TF-IDF with Logistic Regression instead of Deep Learning / BERT?
> **Answer:** For e-commerce deployment with 393,000+ reviews, TF-IDF with Logistic Regression provides:
> 1. High throughput (sub-millisecond inference per review).
> 2. Low resource footprint (1.6 MB total model storage vs. 400+ MB for BERT).
> 3. Direct interpretability through linear regression coefficients.
> 4. Strong 80.16% accuracy and 0.8246 weighted F1.

---

## 📄 License
This project is developed as part of On-the-Job Training (OJT). Dataset courtesy of Amazon Fine Food Reviews on Stanford Network Analysis Project (SNAP) / Kaggle.
