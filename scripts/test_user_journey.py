import io
import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app
from backend.dataset_store import dataset_store, DATASET_FILE

def test_full_user_journey():
    print("="*70)
    print("STARTING END-TO-END USER JOURNEY VERIFICATION")
    print("="*70)
    client = TestClient(app)

    # 1. Reset dataset to empty state
    print("\n1. Resetting dataset to clean empty state...")
    res = client.post("/api/dataset/clear")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    summary = res.json()
    assert summary["has_data"] is False, "Expected has_data to be False"
    assert summary["total_reviews"] == 0, f"Expected 0 reviews, got {summary['total_reviews']}"
    print("   ✓ Empty state verified. Total reviews: 0")

    # 2. Upload CSV dataset
    print("\n2. Uploading sample customer reviews CSV (mode: replace)...")
    sample_csv = (
        'Id,Score,ProfileName,ProductId,Text\n'
        '1,5,Sarah Jenkins,Almonds,"The organic dark chocolate almonds were fresh, crunchy, and delicious!"\n'
        '2,1,David Miller,Chips,"Horrible experience. The bag was torn open and chips tasted stale and awful."\n'
        '3,3,Elena Rostova,Tea,"The tea flavor is okay, not bad but nothing exceptional. A bit too mild."\n'
        '4,5,Marcus Chen,Cookies,"These ginger snap cookies are sensational! Perfect snap and flavor."\n'
        '5,2,Amanda White,PeanutButter,"Disappointed with the texture. It felt like rubber with no peanut taste."\n'
    )
    files = {"file": ("customer_reviews_batch.csv", io.BytesIO(sample_csv.encode("utf-8")), "text/csv")}
    upload_res = client.post("/api/upload-csv", files=files, data={"mode": "replace"})
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    upload_data = upload_res.json()
    assert upload_data["total_reviews"] == 5, f"Expected 5, got {upload_data['total_reviews']}"
    assert upload_data["summary"]["has_data"] is True
    print(f"   ✓ CSV successfully processed {upload_data['total_reviews']} reviews")
    print(f"     Detected text column: {upload_data['text_column_detected']}")
    print(f"     Sentiment breakdown: {upload_data['sentiment_distribution']}")

    # 3. View dashboard metrics
    print("\n3. Fetching updated dashboard summary...")
    dash_res = client.get("/api/dataset/summary")
    assert dash_res.status_code == 200
    dash_data = dash_res.json()
    assert dash_data["total_reviews"] == 5
    assert dash_data["sentiment_distribution"]["Positive"] >= 2
    assert dash_data["sentiment_distribution"]["Negative"] >= 1
    assert len(dash_data["positive_themes"]) > 0
    assert len(dash_data["negative_themes"]) > 0
    print("   ✓ Dashboard metrics confirmed:")
    print(f"     Total: {dash_data['total_reviews']} | Pos: {dash_data['sentiment_distribution']['Positive']} | Neu: {dash_data['sentiment_distribution']['Neutral']} | Neg: {dash_data['sentiment_distribution']['Negative']}")
    print(f"     Positive Themes: {[t['theme'] for t in dash_data['positive_themes'][:3]]}")
    print(f"     Negative Themes: {[t['theme'] for t in dash_data['negative_themes'][:3]]}")

    # 4. Add a new manual review (Positive)
    print("\n4. Submitting a new manual customer review (Positive)...")
    new_rev_1 = {
        "text": "Outstanding honey! Beautiful golden color, rich floral taste, and fast delivery.",
        "rating": 5,
        "customer_name": "Oliver Twist",
        "product_name": "Raw Wildflower Honey"
    }
    sub_res_1 = client.post("/api/reviews", json=new_rev_1)
    assert sub_res_1.status_code == 200, f"Submission failed: {sub_res_1.text}"
    sub_data_1 = sub_res_1.json()
    rev_1 = sub_data_1["review"]
    assert rev_1["predicted_sentiment"] == "Positive", f"Expected Positive, got {rev_1['predicted_sentiment']}"
    assert rev_1["rating"] == 5
    assert rev_1["customer_name"] == "Oliver Twist"
    assert sub_data_1["summary"]["total_reviews"] == 6
    print(f"   ✓ Review analyzed as: {rev_1['predicted_sentiment']} ({rev_1['confidence']*100:.1f}% confidence)")
    print(f"   ✓ Dashboard auto-updated: Total reviews now {sub_data_1['summary']['total_reviews']}")

    # 5. Add a new manual review (Negative with negation word preservation)
    print("\n5. Submitting a second manual customer review (Negative with negation)...")
    new_rev_2 = {
        "text": "Not good at all. I would not buy this again and do not recommend it.",
        "rating": 1,
        "customer_name": "Gordon Gekko",
        "product_name": "Protein Bar"
    }
    sub_res_2 = client.post("/api/reviews", json=new_rev_2)
    assert sub_res_2.status_code == 200, f"Submission failed: {sub_res_2.text}"
    sub_data_2 = sub_res_2.json()
    rev_2 = sub_data_2["review"]
    assert rev_2["predicted_sentiment"] == "Negative", f"Expected Negative, got {rev_2['predicted_sentiment']}"
    assert rev_2["rating"] == 1
    assert sub_data_2["summary"]["total_reviews"] == 7
    print(f"   ✓ Review analyzed as: {rev_2['predicted_sentiment']} ({rev_2['confidence']*100:.1f}% confidence)")
    print(f"   ✓ Dashboard auto-updated: Total reviews now {sub_data_2['summary']['total_reviews']}")

    # 6. Verify persistent storage on disk
    print("\n6. Verifying persistent storage in data/active_reviews.csv...")
    assert os.path.exists(DATASET_FILE), f"File {DATASET_FILE} does not exist"
    df_disk = pd.read_csv(DATASET_FILE)
    assert len(df_disk) == 7, f"Expected 7 rows on disk, found {len(df_disk)}"
    assert "Oliver Twist" in df_disk["customer_name"].values
    assert "Gordon Gekko" in df_disk["customer_name"].values
    print(f"   ✓ Confirmed {len(df_disk)} records permanently saved in {DATASET_FILE}")

    # 7. Test Review Explorer search & filters
    print("\n7. Testing Review Explorer search and filter queries...")
    search_res = client.get("/api/reviews?search=honey")
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total"] == 1
    assert "honey" in search_data["items"][0]["text"].lower()
    print("   ✓ Keyword search for 'honey' returned matching record")

    filter_res = client.get("/api/reviews?sentiment=Negative")
    assert filter_res.status_code == 200
    filter_data = filter_res.json()
    assert filter_data["total"] >= 2
    for item in filter_data["items"]:
        assert item["predicted_sentiment"] == "Negative"
    print(f"   ✓ Sentiment filter for 'Negative' returned {filter_data['total']} reviews")

    # 8. Test CSV Export
    print("\n8. Testing active dataset CSV export...")
    export_res = client.get("/api/dataset/export")
    assert export_res.status_code == 200
    assert "text/csv" in export_res.headers["content-type"]
    exported_csv = export_res.text
    assert "Oliver Twist" in exported_csv
    assert "predicted_sentiment" in exported_csv
    assert "confidence" in exported_csv
    print(f"   ✓ Export generated successfully ({len(exported_csv)} bytes)")

    print("\n" + "="*70)
    print("ALL 8 USER JOURNEY STEPS VERIFIED 100% WORKING AND CONFIRMED!")
    print("="*70)

if __name__ == "__main__":
    test_full_user_journey()
