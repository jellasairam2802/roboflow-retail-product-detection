# Retail product detection with Roboflow

I trained an RF-DETR (Small) object detection model in Roboflow on 200 labeled images of 15 snack and drink products, then ran it on 30 images and analyzed the detections in Python.

## Model results (16-image test set)
- mAP@50: 86.4%
- Precision: 81.2%
- Recall: 77.1%
- F1: 78.7%
- Latency: about 57 ms per image

The test set is very small, so treat these as early estimates. Most images are single products photographed up close, not full shelves, so this is a product-detection prototype and not a shelf-audit tool.

## What the analysis showed
- 36 detections across 14 product classes in 30 images
- Highest average confidence: Aquafina (0.91), Gatorade (0.86)
- Lowest average confidence: KurleezMM (0.44), LaysFC (0.58)
- Precision is higher than recall, so the model is more likely to miss products than invent them

## Sales ops angle
A detection log like this could feed a customer-health view: images processed per day, detections per image, and the share of low-confidence detections. Drops in usage or rising low-confidence rates could flag accounts that need attention.

The over-time chart uses simulated timestamps to show the idea. It is not real usage data.

## Files
- `run_model.py`: runs the trained model on images and saves `predictions.csv`
- `make_charts.py`: builds `dashboard.png` and prints summary stats
- `dashboard.png`: the charts
- `predictions.csv`: raw detections

## Next steps
Test on real shelf photos, add more examples for the weak classes, and replace simulated timestamps with real API usage logs.
