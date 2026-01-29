# -*- coding: utf-8 -*-
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

print("=" * 80)
print("[Part 1] CNN Feature Extractor")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cnn_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
cnn_model = cnn_model.to(device)
cnn_model.eval()

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

feature_extractor = FeatureExtractor(cnn_model).to(device)
feature_extractor.eval()

print("Done: CNN Feature Extractor (1280 dims)")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\n" + "=" * 80)
print("[Part 2] Extract CNN Features")
print("=" * 80)

IMAGE_FOLDER = "1769494106663-shelter_images"
all_imgs = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(".jpg")])
print(f"Total images: {len(all_imgs)}")

cnn_features_list = []

for idx, img_name in enumerate(all_imgs):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    try:
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = feature_extractor(image_tensor)
            features = features.cpu().numpy().flatten()

        cnn_features_list.append(features)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(all_imgs)} done")
    except Exception as e:
        print(f"  Error: {img_name}")

X_cnn = np.array(cnn_features_list)
print(f"\nCNN features shape: {X_cnn.shape}")

print("\n" + "=" * 80)
print("[Part 3] Table Data")
print("=" * 80)

n_samples = len(cnn_features_list)
np.random.seed(42)

X_table = np.column_stack([
    np.random.randint(0, 2, n_samples),
    np.random.uniform(0.5, 15, n_samples),
    np.random.randint(0, 4, n_samples),
    np.random.uniform(2, 40, n_samples),
    np.random.randint(-1, 2, n_samples),
    np.random.randint(0, 2, n_samples),
    np.random.randint(0, 10, n_samples),
    np.random.randint(0, 10, n_samples),
])

table_features = ['is_mixed', 'age_years', 'sex_neutered', 'weight_kg',
                  'health_score', 'has_attack', 'care_encoded', 'org_encoded']

y = np.random.randint(0, 3, n_samples)

print(f"Table data shape: {X_table.shape}")

print("\n" + "=" * 80)
print("[Part 4] Combine Features")
print("=" * 80)

scaler = StandardScaler()
X_table_scaled = scaler.fit_transform(X_table)

X_combined = np.concatenate([X_cnn, X_table_scaled], axis=1)

print(f"CNN features:   {X_cnn.shape[1]} dims")
print(f"Table data:     {X_table_scaled.shape[1]} dims")
print(f"Total features: {X_combined.shape[1]} dims")

print("\n" + "=" * 80)
print("[Part 5] Train XGBoost")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

model_multimodal = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='mlogloss'
)

print("Training...")
model_multimodal.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = model_multimodal.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

print("\n" + "=" * 80)
print("[Part 6] Save Model")
print("=" * 80)

feature_names = [f'CNN_{i}' for i in range(X_cnn.shape[1])] + table_features

model_data = {
    'model': model_multimodal,
    'feature_extractor': feature_extractor,
    'scaler': scaler,
    'feature_names': feature_names,
    'available_features': table_features,
    'test_accuracy': test_acc
}

with open('multimodal_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Saved: multimodal_model.pkl")
print("\nStructure:")
print("  - model: XGBoost classifier")
print("  - feature_extractor: CNN (1280 dims)")
print("  - scaler: StandardScaler")

print("\n" + "=" * 80)
print("[Verify]")
print("=" * 80)

with open('multimodal_model.pkl', 'rb') as f:
    loaded = pickle.load(f)

print("Keys:", list(loaded.keys()))
print(f"model: {type(loaded['model'])}")
print(f"feature_extractor: {type(loaded['feature_extractor'])}")
print(f"scaler: {type(loaded['scaler'])}")
print(f"Has predict_proba: {hasattr(loaded['model'], 'predict_proba')}")

print("\nDone!")
