import sys
import os
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import xgboost as xgb

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

def test_prediction():
    print(f"Loading model from {MODEL_PKL}")
    if not os.path.exists(MODEL_PKL):
        print("Model file not found!")
        return

    try:
        with open(MODEL_PKL, "rb") as f:
            bundle = pickle.load(f)
        print("Bundle loaded successfully.")
    except Exception as e:
        print(f"Failed to load bundle: {e}")
        return

    model = bundle.get("model")
    feature_extractor = bundle.get("feature_extractor")
    device = torch.device("cpu")
    scaler = bundle.get("scaler")

    print(f"Model type: {type(model)}")
    # print(f"Feature Extractor: {feature_extractor}")
    print(f"Scaler: {scaler}")

    if feature_extractor is None:
        print("Feature extractor is None!")
        # Try to restore it if missing (simulating logic that might be needed)
        # But for reproduction, we just want to see if it fails.
    
    # Create dummy image
    image_pil = Image.new("RGB", (300, 300), color="white")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    # 1. Extract features
    try:
        if feature_extractor:
            feature_extractor = feature_extractor.to(device)
            feature_extractor.eval()
            with torch.no_grad():
                img_features = feature_extractor(img_tensor).cpu().numpy().flatten()
            print(f"Image features shape: {img_features.shape}")
        else:
            print("Skipping image features (extractor missing)")
            img_features = np.zeros(1280) 
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return

    # 2. Tabular features
    tabular_list = [0, 3, 0, 10.0, 0, 0, 0, 0] # dummy data
    if scaler:
        tabular_features = scaler.transform([tabular_list])[0]
    else:
        tabular_features = np.array(tabular_list)
    print(f"Tabular features shape: {tabular_features.shape}")

    # 3. Concatenate
    final_features = np.concatenate([img_features, tabular_features]).reshape(1, -1)
    print(f"Final features shape: {final_features.shape}")

    # 4. Predict
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(final_features)[0]
            print(f"Prediction (proba): {probs}")
        else:
            pred = model.predict(final_features)[0]
            print(f"Prediction (class): {pred}")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    test_prediction()
