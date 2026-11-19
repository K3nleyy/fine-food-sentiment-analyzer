import io
import unittest
from fastapi.testclient import TestClient
from backend.main import app
from backend.dataset_store import dataset_store

class TestSentimentAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Reset dataset before each test
        dataset_store.clear()

    def test_01_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("platform", data)
        self.assertEqual(data["platform"], "Fine-Food Insights")

    def test_02_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["classes"], ["Negative", "Neutral", "Positive"])

    def test_03_dataset_summary_empty(self):
        response = self.client.get("/api/dataset/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["has_data"])
        self.assertEqual(data["total_reviews"], 0)
        self.assertEqual(data["sentiment_distribution"]["Positive"], 0)

    def test_04_submit_review_manual(self):
        payload = {
            "text": "The dark chocolate almonds were fresh, crunchy, and absolutely delicious!",
            "rating": 5,
            "customer_name": "Sarah Connor",
            "product_name": "Dark Chocolate Almonds"
        }
        response = self.client.post("/api/reviews", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify review item
        review = data["review"]
        self.assertEqual(review["predicted_sentiment"], "Positive")
        self.assertEqual(review["rating"], 5)
        self.assertEqual(review["customer_name"], "Sarah Connor")
        self.assertEqual(review["product_name"], "Dark Chocolate Almonds")
        self.assertGreater(review["confidence"], 0.70)

        # Verify summary auto-updated
        summary = data["summary"]
        self.assertTrue(summary["has_data"])
        self.assertEqual(summary["total_reviews"], 1)
        self.assertEqual(summary["sentiment_distribution"]["Positive"], 1)
        self.assertEqual(summary["sentiment_percentages"]["Positive"], 100.0)

    def test_05_submit_negative_review_rating_stored_separately(self):
        payload = {
            "text": "Terrible taste, arrived completely stale and bag was torn open. Horrible!",
            "rating": 1,
            "customer_name": "John Doe",
            "product_name": "Potato Chips"
        }
        response = self.client.post("/api/reviews", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        review = data["review"]
        
        # Model predicted sentiment is Negative
        self.assertEqual(review["predicted_sentiment"], "Negative")
        # Rating is stored distinctly as integer 1
        self.assertEqual(review["rating"], 1)

    def test_06_list_reviews_pagination_and_search(self):
        # Add 3 reviews
        self.client.post("/api/reviews", json={"text": "Fresh roasted coffee beans with wonderful aroma.", "rating": 5, "product_name": "Coffee"})
        self.client.post("/api/reviews", json={"text": "Stale chips, arrived crushed.", "rating": 1, "product_name": "Chips"})
        self.client.post("/api/reviews", json={"text": "Decent tea, average flavor.", "rating": 3, "product_name": "Tea"})

        # Test listing
        res = self.client.get("/api/reviews?page=1&limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 3)
        self.assertEqual(len(data["items"]), 3)

        # Test search filter
        search_res = self.client.get("/api/reviews?search=coffee")
        self.assertEqual(search_res.status_code, 200)
        search_data = search_res.json()
        self.assertEqual(search_data["total"], 1)
        self.assertIn("coffee", search_data["items"][0]["text"].lower())

        # Test sentiment filter
        neg_res = self.client.get("/api/reviews?sentiment=Negative")
        self.assertEqual(neg_res.status_code, 200)
        neg_data = neg_res.json()
        self.assertEqual(neg_data["total"], 1)
        self.assertEqual(neg_data["items"][0]["predicted_sentiment"], "Negative")

    def test_07_upload_csv_replace_and_append_modes(self):
        csv_content_1 = (
            'Id,Score,Text\n'
            '1,5,"Delicious cookies, my whole family loved them!"\n'
            '2,1,"Disgusting and stale. Never purchasing again."\n'
        )
        files = {
            "file": ("batch1.csv", io.BytesIO(csv_content_1.encode("utf-8")), "text/csv")
        }
        # Upload with replace
        res1 = self.client.post("/api/upload-csv", files=files, data={"mode": "replace"})
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1["total_reviews"], 2)
        self.assertEqual(data1["summary"]["total_reviews"], 2)

        # Upload second file with append
        csv_content_2 = (
            'Id,Score,Text\n'
            '3,3,"Average flavor, not bad but nothing exceptional."\n'
        )
        files2 = {
            "file": ("batch2.csv", io.BytesIO(csv_content_2.encode("utf-8")), "text/csv")
        }
        res2 = self.client.post("/api/upload-csv", files=files2, data={"mode": "append"})
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2["summary"]["total_reviews"], 3)
        self.assertEqual(data2["summary"]["sentiment_distribution"]["Positive"], 1)
        self.assertEqual(data2["summary"]["sentiment_distribution"]["Negative"], 1)
        self.assertEqual(data2["summary"]["sentiment_distribution"]["Neutral"], 1)

    def test_08_load_sample_dataset_and_clear(self):
        # Load sample
        res_sample = self.client.post("/api/dataset/sample")
        self.assertEqual(res_sample.status_code, 200)
        summary = res_sample.json()
        self.assertTrue(summary["has_data"])
        self.assertEqual(summary["total_reviews"], 25)
        self.assertGreater(summary["sentiment_distribution"]["Positive"], 0)
        self.assertGreater(summary["sentiment_distribution"]["Negative"], 0)
        self.assertGreater(summary["sentiment_distribution"]["Neutral"], 0)
        self.assertGreater(len(summary["positive_themes"]), 0)
        self.assertGreater(len(summary["negative_themes"]), 0)

        # Clear dataset
        res_clear = self.client.post("/api/dataset/clear")
        self.assertEqual(res_clear.status_code, 200)
        cleared_summary = res_clear.json()
        self.assertFalse(cleared_summary["has_data"])
        self.assertEqual(cleared_summary["total_reviews"], 0)

    def test_09_export_dataset(self):
        # Add a review and export
        self.client.post("/api/reviews", json={"text": "Terrific coffee beans!", "rating": 5})
        res = self.client.get("/api/dataset/export")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/csv", res.headers["content-type"])
        content = res.text
        self.assertIn("Terrific coffee beans!", content)
        self.assertIn("predicted_sentiment", content)

    def test_10_stateless_predict(self):
        res = self.client.post("/predict", json={"text": "Fresh and crunchy!"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["sentiment"], "Positive")

if __name__ == "__main__":
    unittest.main()
