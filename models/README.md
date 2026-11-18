# Trained model weights

Training writes checkpoints here (they are gitignored):

| File | Produced by | Used by |
|------|-------------|---------|
| `ingredient_classifier.pt` | `python -m train --task classifier` | `FoodRecognizer`, detector stage 2 |
| `food_detector.pt` | `python -m train --task detector` | `FoodDetector` stage 1 (bounding boxes) |

If a checkpoint is present, the Streamlit app loads it instead of the default ImageNet/COCO heads. Delete a file to revert to pretrained weights.