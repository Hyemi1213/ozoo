import pickle

with open('multimodal_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

print('=== multimodal_model.pkl 구조 ===')
print('Keys:', list(bundle.keys()))

for key in bundle.keys():
    val = bundle.get(key)
    print(f'\n{key}:')
    print(f'  Type: {type(val)}')
    if val is not None and hasattr(val, 'predict_proba'):
        print(f'  Has predict_proba: True (XGBoost)')
    if val is not None and hasattr(val, 'forward'):
        print(f'  Has forward: True (PyTorch)')
