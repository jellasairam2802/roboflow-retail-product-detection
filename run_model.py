import os, csv, datetime
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
).configure(InferenceConfiguration(api_key_transport="header"))

MODEL_ID = "retail-shelf-stock/1"

rows = []
files = sorted(os.listdir("images"))[:30]
for i, fname in enumerate(files):
    result = client.infer(os.path.join("images", fname), model_id=MODEL_ID)
    ts = datetime.datetime(2026, 10, 1) + datetime.timedelta(hours=6 * i)
    for p in result["predictions"]:
        rows.append([fname, ts.isoformat(), p["class"], round(p["confidence"], 3)])
    print(fname, len(result["predictions"]), "detections")

with open("predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "timestamp", "class", "confidence"])
    w.writerows(rows)
print("Saved predictions.csv")