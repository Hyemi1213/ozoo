import pickle
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models

# Define FeatureExtractor as in app.py
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

MODEL_PKL = "multimodal_model.pkl"

if not os.path.exists(MODEL_PKL):
    print("Model file not found")
    sys.exit(1)

try:
    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    print("Keys in bundle:", bundle.keys())
    if "feature_extractor" in bundle:
        print("feature_extractor type:", type(bundle["feature_extractor"]))
    else:
        print("feature_extractor key is MISSING")
except Exception as e:
    print("Error loading pickle:", e)
