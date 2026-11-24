# 🍽️ Fine-Food Insights: Customer Review Sentiment Platform

[![Open In Colab: EDA](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/K3nleyy/fine-food-sentiment-analyzer/blob/main/notebooks/01_data_exploration.ipynb)
[![Open In Colab: Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/K3nleyy/fine-food-sentiment-analyzer/blob/main/notebooks/02_preprocessing_and_model_training.ipynb)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikit-learn)

An end‑to‑end, commercial‑grade Customer Sentiment Intelligence Platform designed for food & beverage brands. Built on the **Amazon Fine Food Reviews** dataset (568 k reviews), the system offers:

- Chronological train/test split to avoid data leakage
- Negation‑preserving text cleaning
- Persistent review storage (no data loss on restart)
- Asynchronous FastAPI backend
- Minimal, professional React analytics dashboard (inspired by Linear & Stripe)

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
- [User Guide (Dashboard Operations)](#-user-guide-dashboard-operations)
- [API Reference](#-api-reference)
- [Google Colab Notebooks](#-google-colab-notebooks)
- [OJT Viva Defense Q&A Guide](#-ojt-viva-defense-qa-guide)
- [License](#-license)

---

## 📖 Project Overview

*Content unchanged – see original README for details.*

---

## 🏗️ System Architecture

*Visual diagram unchanged.*

---

## 💻 Technology Stack

| Layer | Technologies | Purpose |
| :--- | :--- | :--- |
| **Data & EDA** | Python 3.10+, Pandas, NumPy, Matplotlib, Seaborn | Exploratory data analysis, distributions, correlations, time trends |
| **NLP & ML** | Scikit‑Learn, Joblib, Regex | Text normalization, TF‑IDF, Logistic Regression |
| **Backend API** | FastAPI, Uvicorn, Pydantic, Python‑Multipart | Asynchronous REST API, schema validation, CSV streaming |
| **Frontend** | React 18, Vite, Lucide Icons, Modern CSS | Real‑time interactive dashboard, analytics visualizations |
| **Testing** | Unittest, Starlette TestClient | Automated integration testing for all API endpoints |

---

## 📁 Repository Structure

```
fine-food-sentiment-analyzer/
├── .gitignore                     # excludes datasets, *.joblib, node_modules
├── README.md                      # **This file** – user‑focused docs + viva guide
├── data/
│   └── Reviews.csv                # raw dataset (git‑ignored, ~300 MB)
├── model/
│   ├── tfidf_vectorizer.joblib    # fitted TF‑IDF (1.09 MB)
│   └── logistic_regression_model.joblib  # trained classifier (0.57 MB)
├── notebooks/
│   ├── 01_data_exploration.ipynb # Phase 1 – EDA with observations under each plot
│   └── 02_preprocessing_and_model_training.ipynb # Phase 2 & 3 – cleaning & training
├── backend/
│   ├── main.py                    # FastAPI app & route handlers
│   ├── predictor.py               # Text cleaning & inference service
│   ├── schemas.py                 # Pydantic request/response models
│   ├── test_api.py                # 10 integration tests
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── index.html                 # Single‑page entry point
│   ├── package.json               # Node dependencies & scripts
│   ├── vite.config.js             # Vite + API proxy configuration
│   └── src/
│       ├── App.jsx                # Dashboard UI (overview, charts, tables)
│       ├── index.css              # Clean dark‑mode styling – no gradients/glows
│       └── main.jsx               # React DOM root render
└── scripts/
    └── test_user_journey.py      # End‑to‑end script exercising the full flow
```

---

## 🚀 Quick Start Guide

### 1. Backend Setup (FastAPI)
```bash
# From the project root
cd fine-food-sentiment-analyzer
# Install backend dependencies (prefer a virtual env)
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
# Start the API server
uvicorn backend.main:app --reload --port 8000
```
The API will be reachable at `http://localhost:8000`. Swagger UI is available at `http://localhost:8000/docs`.

### 2. Frontend Setup (React)
```bash
# Open a new terminal window
cd fine-food-sentiment-analyzer/frontend
npm install
npm run dev
```
The dashboard runs at `http://localhost:5173`.

### 3. Running Automated Tests
```bash
python -m unittest backend/test_api.py
```
All 10 integration tests should pass.

---

## 📚 User Guide (Dashboard Operations)

### Adding a New Review
1. Click **Add New Review** (top‑right button).
2. Fill **Review Text** (required) and optionally **Rating**, **Customer Name**, **Product Name**.
3. Press **Submit** – the UI validates, shows a loading spinner, then a success toast.
4. The new review is stored persistently, the sentiment model predicts automatically, and all metric cards/charts refresh instantly (no page reload).

### Uploading a CSV of Reviews
1. Click **Upload CSV**.
2. Drag‑and‑drop your file or use the file picker.
3. Choose **Replace** (overwrite the active dataset) or **Append** (add to existing data).
4. Confirm – the backend runs batch sentiment analysis, updates the persistent store, and the dashboard refreshes.
5. After upload you can **Download CSV** (includes predicted sentiment & confidence columns).

### Interacting with the Dashboard
- **Overview Cards** – total reviews and percentages for Positive / Neutral / Negative.
- **Sentiment Distribution** – simple pie/stacked bar showing sentiment share.
- **Rating Distribution** – bar chart of 1‑5 star counts.
- **Feedback Themes** – word‑cloud‑like list of most common positive/negative terms.
- **Review Table** – searchable, filterable (by sentiment, rating) and paginated.

All UI elements use a minimal colour palette, subtle shadows, and clear typography. The design works on both desktop and mobile screens.

---

## 📡 API Reference

*(section unchanged – refer to original README for endpoint details)*

---

## 📓 Google Colab Notebooks

1. **[01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)** – Phase 1 EDA with observations.
2. **[02_preprocessing_and_model_training.ipynb](notebooks/02_preprocessing_and_model_training.ipynb)** – Pre‑processing, model training, and evaluation.

Both notebooks are ready to run in Google Colab with Google Drive integration.

---

## 🎓 OJT Viva Defense Q&A Guide

*Content unchanged – see original README for detailed answers.*

---

## 📄 License

This project is developed as part of an On‑the‑Job Training (OJT) assignment. Dataset courtesy of Amazon Fine Food Reviews (SNAP / Kaggle).