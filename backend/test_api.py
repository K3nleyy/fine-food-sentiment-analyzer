import os
import io
import unittest
from fastapi.testclient import TestClient
from backend.main import app

class TestSentimentAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("project", data)
        self.assertIn("endpoints", data)

    def test_02_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["classes"], ["Negative", "Neutral", "Positive"])
        self.assertGreater(data["vocabulary_size"], 1000)

    def test_03_predict_positive(self):
        payload = {"text": "The dark chocolate almonds were fresh, crunchy, and absolutely delicious! Best purchase ever."}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sentiment"], "Positive")
        self.assertGreater(data["confidence"], 0.70)
        self.assertGreater(data["probabilities"]["Positive"], data["probabilities"]["Negative"])

    def test_04_predict_negative(self):
        payload = {"text": "Terrible experience. The bag was torn open, stale chips, and awful taste. Worst ever."}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sentiment"], "Negative")
        self.assertGreater(data["confidence"], 0.70)
        self.assertGreater(data["probabilities"]["Negative"], data["probabilities"]["Positive"])

    def test_05_predict_neutral(self):
        payload = {"text": "The flavor is okay, not bad but nothing special. Average tea."}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sentiment"], "Neutral")

    def test_06_predict_empty_error(self):
        payload = {"text": "   "}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot be empty", response.json()["detail"])

    def test_07_predict_batch(self):
        payload = {
            "reviews": [
                "Loved this delicious treat!",
                "Completely awful and disgusting, threw it in the trash.",
                "It was average and okay, not great."
            ]
        }
        response = self.client.post("/predict-batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_reviews"], 3)
        self.assertEqual(data["sentiment_distribution"]["Positive"], 1)
        self.assertEqual(data["sentiment_distribution"]["Negative"], 1)
        self.assertEqual(data["sentiment_distribution"]["Neutral"], 1)

    def test_08_upload_csv(self):
        csv_content = (
            'Id,ProductId,Text\n'
            '1,B001,"Absolutely loved this product! Fantastic flavor."\n'
            '2,B002,"Terrible quality. Arrived broken and stale."\n'
            '3,B003,"It is alright, nothing special but drinkable."\n'
            '4,B004,"Super yummy snack, will definitely order again."\n'
        )
        files = {
            "file": ("test_reviews.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
        }
        response = self.client.post("/upload-csv", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_reviews"], 4)
        self.assertEqual(data["text_column_detected"], "Text")
        self.assertEqual(data["sentiment_distribution"]["Positive"], 2)
        self.assertEqual(data["sentiment_distribution"]["Negative"], 1)
        self.assertEqual(data["sentiment_distribution"]["Neutral"], 1)
        self.assertEqual(len(data["results_sample"]), 4)

    def test_09_upload_invalid_file_extension(self):
        files = {
            "file": ("image.png", io.BytesIO(b"fake-image-bytes"), "image/png")
        }
        response = self.client.post("/upload-csv", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid file format", response.json()["detail"])

    def test_10_export_csv(self):
        csv_content = (
            "Id,Text\n"
            "1,Delicious coffee.\n"
            "2,Bad taste.\n"
        )
        files = {
            "file": ("export_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
        }
        response = self.client.post("/export-csv", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers["content-type"])
        content_text = response.text
        self.assertIn("Predicted_Sentiment", content_text)
        self.assertIn("Confidence", content_text)
        self.assertIn("Positive", content_text)
        self.assertIn("Negative", content_text)

if __name__ == "__main__":
    unittest.main()
