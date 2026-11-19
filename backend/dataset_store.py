import os
import io
import re
import csv
import time
import uuid
import collections
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from backend.predictor import predictor, clean_text, LABEL_MAP

# Stopwords to filter out when extracting top feedback themes
STOPWORDS = {
    'the', 'and', 'a', 'to', 'of', 'is', 'it', 'in', 'i', 'this', 'that', 'for', 'my', 'with', 
    'on', 'you', 'but', 'was', 'are', 'have', 'as', 'they', 'at', 'be', 'so', 'one', 'all', 
    'or', 'from', 'if', 'we', 'an', 'by', 'them', 'these', 'very', 'just', 'more', 'about', 
    'can', 'me', 'had', 'has', 'will', 'do', 'when', 'out', 'up', 'some', 'there', 'what', 
    'which', 'their', 'our', 'than', 'into', 'would', 'other', 'only', 'also', 'get', 'product', 
    'taste', 'flavor', 'food', 'like', 'good', 'bought', 'ordered', 'amazon', 'bag', 'box'
}

DATASET_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "active_reviews.csv"))

COLUMNS = [
    'id', 'text', 'cleaned_text', 'rating', 'customer_name', 'product_name', 
    'predicted_sentiment', 'confidence', 'prob_positive', 'prob_neutral', 
    'prob_negative', 'created_at'
]


class DatasetStore:
    """
    Persistent store for customer reviews and live analytics.
    Maintains `data/active_reviews.csv` for newly submitted reviews
    and uploaded review datasets.
    """
    def __init__(self, filepath: str = DATASET_FILE):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            # Create empty CSV with proper header
            df = pd.DataFrame(columns=COLUMNS)
            df.to_csv(self.filepath, index=False)

    def load_df(self) -> pd.DataFrame:
        self._ensure_file()
        try:
            df = pd.read_csv(self.filepath, dtype={'rating': 'object', 'id': 'str'})
            # Ensure all columns exist
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df
        except Exception:
            return pd.DataFrame(columns=COLUMNS)

    def save_df(self, df: pd.DataFrame):
        self._ensure_file()
        # Atomic write using temporary file to prevent corruption
        temp_file = self.filepath + ".tmp"
        df.to_csv(temp_file, index=False)
        os.replace(temp_file, self.filepath)

    def add_single_review(
        self, 
        text: str, 
        rating: Optional[int] = None, 
        customer_name: Optional[str] = None, 
        product_name: Optional[str] = None
    ) -> dict:
        """
        Validates, runs sentiment prediction, appends review to persistent CSV,
        and returns the created review item and updated summary.
        """
        text = text.strip()
        if not text:
            raise ValueError("Review text cannot be empty.")

        # Run model inference
        prediction = predictor.predict_single(text)
        sentiment = prediction["sentiment"]
        confidence = prediction["confidence"]
        probs = prediction["probabilities"]

        review_id = f"rev_{int(time.time()*1000)}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        new_row = {
            'id': review_id,
            'text': text,
            'cleaned_text': prediction["cleaned_text"],
            'rating': str(rating) if rating is not None and 1 <= rating <= 5 else "",
            'customer_name': customer_name.strip() if customer_name else "Anonymous Customer",
            'product_name': product_name.strip() if product_name else "General Food Item",
            'predicted_sentiment': sentiment,
            'confidence': confidence,
            'prob_positive': probs["Positive"],
            'prob_neutral': probs["Neutral"],
            'prob_negative': probs["Negative"],
            'created_at': timestamp
        }

        df = self.load_df()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_df(df)

        return new_row

    def import_dataframe(self, incoming_df: pd.DataFrame, mode: str = "replace") -> Tuple[int, str]:
        """
        Takes an uploaded DataFrame, detects relevant columns, predicts sentiment,
        and saves into active_reviews.csv.
        mode: 'replace' to start fresh, or 'append' to add to existing reviews.
        """
        if incoming_df.empty:
            raise ValueError("The uploaded CSV contains no records.")

        # Detect text column
        text_col = None
        for col in ['Text', 'text', 'Review', 'review', 'Reviews', 'reviews', 'Comment', 'comment', 'Summary', 'summary', 'body']:
            if col in incoming_df.columns:
                text_col = col
                break
        if not text_col:
            for col in incoming_df.columns:
                if incoming_df[col].dtype == 'object':
                    text_col = col
                    break
        if not text_col:
            raise ValueError(f"Could not identify review text column. Found columns: {list(incoming_df.columns)}")

        # Detect optional rating column
        rating_col = None
        for col in ['Score', 'score', 'Rating', 'rating', 'Stars', 'stars']:
            if col in incoming_df.columns:
                rating_col = col
                break

        # Detect optional customer/user column
        customer_col = None
        for col in ['ProfileName', 'CustomerName', 'customer_name', 'UserName', 'User', 'user']:
            if col in incoming_df.columns:
                customer_col = col
                break

        # Detect optional product column
        product_col = None
        for col in ['ProductId', 'ProductName', 'product_name', 'product', 'Product']:
            if col in incoming_df.columns:
                product_col = col
                break

        # Detect optional date column
        date_col = None
        for col in ['Time', 'Date', 'date', 'review_date', 'timestamp', 'CreatedAt']:
            if col in incoming_df.columns:
                date_col = col
                break

        raw_texts = incoming_df[text_col].fillna("").astype(str).tolist()
        batch_out = predictor.predict_batch(raw_texts)

        records = []
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        for idx, item in enumerate(batch_out["results"]):
            row_id = str(incoming_df.iloc[idx].get('Id', f"csv_{idx+1}"))
            
            # Rating handling
            raw_rating = ""
            if rating_col:
                val = incoming_df.iloc[idx][rating_col]
                try:
                    val_int = int(float(val))
                    if 1 <= val_int <= 5:
                        raw_rating = str(val_int)
                except Exception:
                    pass

            # Customer name
            cust_name = "Customer"
            if customer_col:
                cval = str(incoming_df.iloc[idx][customer_col]).strip()
                if cval and cval != "nan":
                    cust_name = cval

            # Product name
            prod_name = "Food Product"
            if product_col:
                pval = str(incoming_df.iloc[idx][product_col]).strip()
                if pval and pval != "nan":
                    prod_name = pval

            # Date handling
            created_at = now_str
            if date_col:
                dval = incoming_df.iloc[idx][date_col]
                try:
                    if str(dval).isdigit():
                        created_at = datetime.utcfromtimestamp(int(dval)).strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        parsed = pd.to_datetime(dval)
                        created_at = parsed.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

            records.append({
                'id': row_id,
                'text': item["text"],
                'cleaned_text': item["cleaned_text"],
                'rating': raw_rating,
                'customer_name': cust_name,
                'product_name': prod_name,
                'predicted_sentiment': item["sentiment"],
                'confidence': item["confidence"],
                'prob_positive': item["probabilities"]["Positive"],
                'prob_neutral': item["probabilities"]["Neutral"],
                'prob_negative': item["probabilities"]["Negative"],
                'created_at': created_at
            })

        new_df = pd.DataFrame(records)

        if mode == "append":
            current_df = self.load_df()
            final_df = pd.concat([current_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        self.save_df(final_df)
        return len(new_df), text_col

    def get_summary(self) -> dict:
        """
        Computes business-friendly analytics for the active dataset:
        - Total reviews
        - Positive, Neutral, Negative counts & percentages
        - Rating distribution (1 to 5 stars)
        - Monthly sentiment trends
        - Key positive & negative customer feedback themes
        """
        df = self.load_df()
        total = len(df)

        if total == 0:
            return {
                "has_data": False,
                "total_reviews": 0,
                "sentiment_distribution": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "sentiment_percentages": {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0},
                "rating_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "unrated": 0},
                "timeline": [],
                "positive_themes": [],
                "negative_themes": [],
                "last_updated": None
            }

        # Sentiment counts
        sent_counts = df['predicted_sentiment'].value_counts()
        pos_cnt = int(sent_counts.get("Positive", 0))
        neu_cnt = int(sent_counts.get("Neutral", 0))
        neg_cnt = int(sent_counts.get("Negative", 0))

        pos_pct = round(pos_cnt / total * 100, 1) if total > 0 else 0.0
        neu_pct = round(neu_cnt / total * 100, 1) if total > 0 else 0.0
        neg_pct = round(neg_cnt / total * 100, 1) if total > 0 else 0.0

        # Rating distribution
        rating_dist = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "unrated": 0}
        for r in df['rating']:
            r_str = str(r).strip()
            if r_str in ["1", "2", "3", "4", "5"]:
                rating_dist[r_str] += 1
            else:
                rating_dist["unrated"] += 1

        # Timeline trend (by year-month)
        timeline = []
        try:
            df_dates = df.copy()
            df_dates['dt'] = pd.to_datetime(df_dates['created_at'], errors='coerce')
            df_dates = df_dates.dropna(subset=['dt'])
            if not df_dates.empty:
                df_dates['period'] = df_dates['dt'].dt.to_period('M').astype(str)
                grouped = df_dates.groupby(['period', 'predicted_sentiment']).size().unstack(fill_value=0)
                # Keep up to the most recent 12 periods
                recent_periods = grouped.tail(12)
                for period, row in recent_periods.iterrows():
                    timeline.append({
                        "period": str(period),
                        "Positive": int(row.get("Positive", 0)),
                        "Neutral": int(row.get("Neutral", 0)),
                        "Negative": int(row.get("Negative", 0))
                    })
        except Exception:
            timeline = []

        # Extract top feedback themes from actual review texts
        pos_texts = df[df['predicted_sentiment'] == 'Positive']['cleaned_text'].dropna().tolist()
        neg_texts = df[df['predicted_sentiment'] == 'Negative']['cleaned_text'].dropna().tolist()

        pos_themes = self._extract_themes(pos_texts, top_n=6)
        neg_themes = self._extract_themes(neg_texts, top_n=6)

        last_updated = df['created_at'].max() if 'created_at' in df.columns and not df.empty else None

        return {
            "has_data": True,
            "total_reviews": total,
            "sentiment_distribution": {
                "Positive": pos_cnt,
                "Neutral": neu_cnt,
                "Negative": neg_cnt
            },
            "sentiment_percentages": {
                "Positive": pos_pct,
                "Neutral": neu_pct,
                "Negative": neg_pct
            },
            "rating_distribution": rating_dist,
            "timeline": timeline,
            "positive_themes": pos_themes,
            "negative_themes": neg_themes,
            "last_updated": last_updated
        }

    def _extract_themes(self, texts: List[str], top_n: int = 6) -> List[Dict[str, Any]]:
        """
        Extracts meaningful repeated phrases and descriptor keywords from customer texts.
        """
        words = []
        for t in texts:
            if not isinstance(t, str):
                continue
            tokens = [w for w in t.split() if len(w) > 3 and w not in STOPWORDS]
            words.extend(tokens)
            # Bigrams
            for i in range(len(tokens) - 1):
                words.append(f"{tokens[i]} {tokens[i+1]}")

        counts = collections.Counter(words)
        return [{"theme": theme, "mentions": count} for theme, count in counts.most_common(top_n)]

    def get_reviews(
        self, 
        page: int = 1, 
        limit: int = 20, 
        search: str = "", 
        sentiment: str = "ALL", 
        rating: str = "ALL"
    ) -> dict:
        """
        Returns paginated and filtered reviews from the active dataset.
        """
        df = self.load_df()
        if df.empty:
            return {"total": 0, "page": page, "pages": 0, "items": []}

        # Apply sentiment filter
        if sentiment and sentiment != "ALL":
            df = df[df['predicted_sentiment'].str.lower() == sentiment.lower()]

        # Apply rating filter
        if rating and rating != "ALL":
            df = df[df['rating'].astype(str) == str(rating)]

        # Apply text search filter
        if search and search.strip():
            query = search.strip().lower()
            mask = (
                df['text'].str.lower().str.contains(query, na=False) |
                df['customer_name'].str.lower().str.contains(query, na=False) |
                df['product_name'].str.lower().str.contains(query, na=False) |
                df['predicted_sentiment'].str.lower().str.contains(query, na=False)
            )
            df = df[mask]

        total_matching = len(df)
        total_pages = max(1, (total_matching + limit - 1) // limit)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        sliced = df.iloc[start_idx:end_idx]

        items = []
        for _, row in sliced.iterrows():
            items.append({
                "id": str(row["id"]),
                "text": str(row["text"]),
                "rating": int(row["rating"]) if str(row["rating"]).isdigit() else None,
                "customer_name": str(row["customer_name"]) if row["customer_name"] else "Anonymous",
                "product_name": str(row["product_name"]) if row["product_name"] else "Food Product",
                "predicted_sentiment": str(row["predicted_sentiment"]),
                "confidence": round(float(row["confidence"]), 4) if pd.notnull(row["confidence"]) else 0.8,
                "probabilities": {
                    "Positive": round(float(row.get("prob_positive", 0.0)), 4),
                    "Neutral": round(float(row.get("prob_neutral", 0.0)), 4),
                    "Negative": round(float(row.get("prob_negative", 0.0)), 4)
                },
                "created_at": str(row["created_at"]) if row["created_at"] else ""
            })

        return {
            "total": total_matching,
            "page": page,
            "pages": total_pages,
            "items": items
        }

    def clear(self):
        """Empties the active review dataset."""
        df = pd.DataFrame(columns=COLUMNS)
        self.save_df(df)

    def load_sample_dataset(self) -> int:
        """
        Loads a curated, representative set of 25 fine-food customer reviews
        covering positive, neutral, and negative experiences across diverse ratings.
        """
        sample_data = [
            ("The organic dark chocolate almonds were fresh, crunchy, and absolutely delicious! Best snack I have purchased.", 5, "Sarah Jenkins", "Organic Dark Chocolate Almonds", "2024-03-01 10:15:00"),
            ("Horrible experience. The bag was torn open, stale chips everywhere, and smelled like chemicals. Never again.", 1, "David Miller", "Kettle Sea Salt Potato Chips", "2024-03-02 11:30:00"),
            ("The tea flavor is okay, not bad but nothing exceptional. A bit too mild for my personal taste.", 3, "Elena Rostova", "Earl Grey Black Tea", "2024-03-03 09:20:00"),
            ("These gluten-free ginger snap cookies are sensational! Perfect snap and genuine ginger spice.", 5, "Marcus Chen", "Gluten-Free Ginger Snaps", "2024-03-04 14:45:00"),
            ("Disappointed with the texture. It felt like chewing rubber and lacked any real peanut flavor.", 2, "Amanda White", "Creamy All-Natural Peanut Butter", "2024-03-05 16:10:00"),
            ("Decent daily coffee beans. Price is very fair for a 2lb bag, though roast is slightly uneven.", 3, "Tom Bradley", "Medium Roast Colombian Beans", "2024-03-06 08:05:00"),
            ("Arrived completely shattered inside the shipping box. Glass shards mixed into the honey. Total mess.", 1, "Rachel Green", "Wildflower Raw Honey 16oz", "2024-03-07 13:25:00"),
            ("Incredible authentic flavor! Added this virgin olive oil to caprese salad and the freshness stood out.", 5, "Giovanni Rossi", "Extra Virgin Cold-Pressed Olive Oil", "2024-03-08 18:40:00"),
            ("Too salty for my liking. Had high hopes given the reviews, but my family could not finish the bowl.", 2, "Lisa Simpson", "Himalayan Pink Salt Popcorn", "2024-03-09 19:15:00"),
            ("Smooth matcha powder with vibrant green color and no clumps. Blends beautifully with oat milk.", 5, "Kenji Sato", "Ceremonial Grade Matcha Powder", "2024-03-10 07:50:00"),
            ("Not good at all. I would not buy this again and do not recommend it to anyone.", 1, "Robert Hayes", "Protein Energy Bars Variety Pack", "2024-03-11 12:00:00"),
            ("Average granola. Lots of rolled oats but very few almonds or raisins as shown on the packaging.", 3, "Claire Dunphy", "Honey Almond Crunch Granola", "2024-03-12 09:30:00"),
            ("My children adore these fruit gummies! No artificial dyes and pure fruit puree.", 5, "Jessica Day", "Organic Strawberry Fruit Snacks", "2024-03-13 15:20:00"),
            ("Tasted stale and rancid. Checked the expiration date and it was only 2 weeks away from expiring.", 1, "Anthony Hopkins", "Roasted Cashew Halves", "2024-03-14 17:05:00"),
            ("The soup broth is alright, convenience is great for office lunches, but could use more herbs.", 3, "Arthur Dent", "Instant Miso Noodle Soup", "2024-03-15 12:40:00"),
            ("Exceptional balsamic vinegar. Deep, complex, and syrupy sweetness without overwhelming acidity.", 5, "Sophia Loren", "Aged Modena Balsamic Vinegar", "2024-03-16 20:10:00"),
            ("Box was crushed upon arrival and several pods were punctured, spilling coffee grounds.", 1, "Jim Halpert", "Espresso Roast Coffee Pods", "2024-03-17 08:35:00"),
            ("Mild salsa with good tomato chunks. Could use a bit more lime and cilantro, but solid.", 3, "Carlos Santana", "Chunky Medium Garden Salsa", "2024-03-18 18:25:00"),
            ("The sourdough pretzels are top notch! Perfect crunch and authentic fermented tang.", 5, "Heidi Klum", "Artisan Sourdough Hard Pretzels", "2024-03-19 14:15:00"),
            ("Way too sweet. The artificial sweetener aftertaste lingers for hours.", 2, "Gordon Ramsay", "Sugar-Free Vanilla Syrup", "2024-03-20 16:50:00"),
            ("Super fresh walnuts, vacuum sealed perfectly. Great for baking banana bread.", 5, "Martha Stewart", "Raw English Walnut Halves", "2024-03-21 11:10:00"),
            ("Flavor is fine but portion sizes in each pouch are surprisingly small for the price.", 3, "Peter Parker", "Dried Mango Slices 6-Pack", "2024-03-22 13:05:00"),
            ("Awful taste, tasted like cardboard with a sprinkle of seasoning. Threw the entire pack away.", 1, "Brian Griffin", "Cauliflower Tortilla Chips", "2024-03-23 15:45:00"),
            ("Rich, velvety cocoa powder. Made the most decadent hot chocolate for our winter cabin trip.", 5, "Julia Child", "Dutch-Process Baking Cocoa", "2024-03-24 19:30:00"),
            ("It is acceptable for baking but lacks the aroma of pure Madagascar vanilla.", 3, "Paul Hollywood", "Pure Bourbon Vanilla Extract", "2024-03-25 10:20:00")
        ]

        df_in = pd.DataFrame(sample_data, columns=['Text', 'Score', 'ProfileName', 'ProductId', 'Time'])
        count, _ = self.import_dataframe(df_in, mode="replace")
        return count


# Global dataset store instance
dataset_store = DatasetStore()
