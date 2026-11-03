import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Detections per product class
counts = df["class"].value_counts().sort_values()
counts.plot(kind="barh", ax=axes[0], color="#6C2BD9")
axes[0].set_title("Detections per product")
axes[0].set_xlabel("Count")

# 2. Confidence distribution
axes[1].hist(df["confidence"], bins=15, color="#6C2BD9", edgecolor="white")
axes[1].set_title("Confidence distribution")
axes[1].set_xlabel("Confidence")
axes[1].set_ylabel("Detections")

# 3. Detections over time (simulated timestamps)
daily = df.set_index("timestamp").resample("D").size()
daily.plot(ax=axes[2], marker="o", color="#6C2BD9")
axes[2].set_title("Detections per day (simulated timestamps)")
axes[2].set_ylabel("Detections")

plt.tight_layout()
plt.savefig("dashboard.png", dpi=150)

print("Total detections:", len(df))
print("Images with detections:", df["image"].nunique())
print("Average confidence:", round(df["confidence"].mean(), 3))
print("Low-confidence (<0.5) detections:", int((df["confidence"] < 0.5).sum()))
print("\nDetections per product:")
print(df["class"].value_counts().to_string())
print("\nAverage confidence per product:")
print(df.groupby("class")["confidence"].mean().round(3).sort_values().to_string())
print("Saved dashboard.png")