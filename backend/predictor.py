import os
import re
import html
import joblib
import numpy as np

# Label mapping consistent with training pipeline
LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def clean_text(text: str) -> str:
    """
    Cleans raw review text using the exact preprocessing steps
    established in Phase 2 & 3 of the ML pipeline:
    - HTML unescaping & tag removal
    - URL removal
    - Contraction expansion with negation retention
    - Non-alphabet character filtering
    - Lowercasing and whitespace normalization
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities (e.g. &quot;, &amp;)
    text = html.unescape(text)
    
    # 2. Strip HTML tags (e.g. <br />, <p>)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Strip URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # 4. Standardize contractions while preserving negation tokens
    text = re.sub(r"won't", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"can't", "cannot", text, flags=re.IGNORECASE)
    text = re.sub(r"n't", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"'re", " are", text, flags=re.IGNORECASE)
    text = re.sub(r"'s", " is", text, flags=re.IGNORECASE)
    text = re.sub(r"'d", " would", text, flags=re.IGNORECASE)
    text = re.sub(r"'ll", " will", text, flags=re.IGNORECASE)
    text = re.sub(r"'ve", " have", text, flags=re.IGNORECASE)
    text = re.sub(r"'m", " am", text, flags=re.IGNORECASE)
    
    # 5. Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 6. Lowercase & strip excess whitespace
    return ' '.join(text.lower().split())


class SentimentPredictor:
    """
    Inference service for loading trained TF-IDF vectorizer and 
    Logistic Regression classifier and producing sentiment predictions.
    """
    def __init__(self, model_dir: str = None):
        self.vectorizer = None
        self.model = None
        self.is_loaded = False
        self.model_dir = self._resolve_model_dir(model_dir)
        self.load_artifacts()

    def _resolve_model_dir(self, custom_dir: str = None) -> str:
        if custom_dir and os.path.exists(custom_dir):
            return custom_dir
        
        # Check standard relative paths
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates = [
            os.path.join(base_dir, "model"),
            os.path.join(os.getcwd(), "model"),
            "model",
            "../model"
        ]
        for candidate in candidates:
            vec_check = os.path.join(candidate, "tfidf_vectorizer.joblib")
            model_check = os.path.join(candidate, "logistic_regression_model.joblib")
            if os.path.exists(vec_check) and os.path.exists(model_check):
                return candidate
        
        return os.path.join(base_dir, "model")

    def load_artifacts(self):
        vec_path = os.path.join(self.model_dir, "tfidf_vectorizer.joblib")
        model_path = os.path.join(self.model_dir, "logistic_regression_model.joblib")

        if not os.path.exists(vec_path) or not os.path.exists(model_path):
            self.is_loaded = False
            return

        try:
            self.vectorizer = joblib.load(vec_path)
            self.model = joblib.load(model_path)
            self.is_loaded = True
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"Failed to load model artifacts from {self.model_dir}: {e}")

    def predict_single(self, text: str) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model artifacts are not loaded.")

        cleaned = clean_text(text)
        if not cleaned:
            # Fallback if raw text contains only punctuation or empty
            return {
                "text": text,
                "cleaned_text": "",
                "sentiment": "Neutral",
                "confidence": 0.3333,
                "probabilities": {
                    "Negative": 0.3333,
                    "Neutral": 0.3334,
                    "Positive": 0.3333
                }
            }

        vec = self.vectorizer.transform([cleaned])
        pred_idx = int(self.model.predict(vec)[0])
        probs = self.model.predict_proba(vec)[0]

        return {
            "text": text,
            "cleaned_text": cleaned,
            "sentiment": LABEL_MAP[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": {
                "Negative": round(float(probs[0]), 4),
                "Neutral": round(float(probs[1]), 4),
                "Positive": round(float(probs[2]), 4)
            }
        }

    def predict_batch(self, texts: list[str]) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model artifacts are not loaded.")

        cleaned_list = [clean_text(t) for t in texts]
        
        # Guard for completely empty inputs
        processed_texts = [c if c else "neutral" for c in cleaned_list]
        vecs = self.vectorizer.transform(processed_texts)
        pred_indices = self.model.predict(vecs)
        prob_matrix = self.model.predict_proba(vecs)

        results = []
        counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

        for orig, cleaned, idx, probs in zip(texts, cleaned_list, pred_indices, prob_matrix):
            sentiment = LABEL_MAP[int(idx)]
            counts[sentiment] += 1
            conf = round(float(probs[int(idx)]), 4)
            results.append({
                "text": orig,
                "cleaned_text": cleaned,
                "sentiment": sentiment,
                "confidence": conf,
                "probabilities": {
                    "Negative": round(float(probs[0]), 4),
                    "Neutral": round(float(probs[1]), 4),
                    "Positive": round(float(probs[2]), 4)
                }
            })

        total = len(texts)
        percentages = {
            k: round((v / total * 100), 2) if total > 0 else 0.0
            for k, v in counts.items()
        }

        return {
            "total_reviews": total,
            "sentiment_distribution": counts,
            "sentiment_percentages": percentages,
            "results": results
        }


# Global singleton instance for FastAPI
predictor = SentimentPredictor()
