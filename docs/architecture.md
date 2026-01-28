# 모델 아키텍처

## 전체 구조

```
┌─────────────────┐         ┌──────────────────┐
│   이미지 입력    │         │   표 데이터 입력  │
│  (224x224x3)    │         │    (8 features)  │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌──────────────────┐
│ EfficientNet-B0 │         │  Feature Eng.    │
│  (ImageNet 학습) │         │  - is_mixed      │
│                 │         │  - age_years     │
│  Features:      │         │  - sex_neutered  │
│  ├─ Conv blocks │         │  - weight_kg     │
│  ├─ MBConv      │         │  - health_score  │
│  └─ Avgpool     │         │  - has_attack    │
└────────┬────────┘         │  - care_encoded  │
         │                  │  - org_encoded   │
         │                  └────────┬─────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌──────────────────┐
│ CNN Features    │         │ StandardScaler   │
│  (1280 dims)    │         │   (normalize)    │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         └────────────┬──────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │   Concatenate    │
            │   (1288 dims)    │
            └─────────┬────────┘
                      │
                      ▼
            ┌──────────────────┐
            │    XGBoost       │
            │  Classifier      │
            │                  │
            │  max_depth: 4    │
            │  lr: 0.05        │
            │  n_est: 100      │
            └─────────┬────────┘
                      │
                      ▼
            ┌──────────────────┐
            │   3-class        │
            │   Output         │
            │                  │
            │  0: 생존         │
            │  1: 자연사       │
            │  2: 안락사       │
            └──────────────────┘
```

## 1. CNN 파트 (이미지 분석)

### EfficientNet-B0 선택 이유

**장점:**
- ✅ 경량 (5.3M 파라미터)
- ✅ 높은 정확도
- ✅ 빠른 추론 속도
- ✅ ImageNet 사전학습 가능

**대안 비교:**
| 모델 | 파라미터 | Top-1 Acc | 속도 |
|------|----------|-----------|------|
| EfficientNet-B0 | 5.3M | 77.1% | ⭐⭐⭐⭐⭐ |
| ResNet50 | 25.6M | 76.0% | ⭐⭐⭐ |
| MobileNetV2 | 3.5M | 72.0% | ⭐⭐⭐⭐⭐ |

### CNN 피처 추출

```python
class FeatureExtractor(nn.Module):
    def __init__(self, efficientnet):
        super().__init__()
        self.features = efficientnet.features  # Conv blocks
        self.avgpool = efficientnet.avgpool    # Global pooling
    
    def forward(self, x):
        # Input: (B, 3, 224, 224)
        x = self.features(x)      # (B, 1280, 7, 7)
        x = self.avgpool(x)        # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)    # (B, 1280)
        return x
```

**출력:**
- 1280차원 피처 벡터
- 이미지의 고수준 특징 표현

### Data Augmentation

**훈련 시:**
```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),      # 랜덤 크롭
    transforms.RandomHorizontalFlip(0.5),   # 좌우 반전
    transforms.RandomRotation(20),          # 회전
    transforms.ColorJitter(0.3, 0.3, 0.3),  # 색상 변경
    transforms.RandomAffine(...),           # 아핀 변환
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])
```

**검증/테스트 시:**
```python
transforms.Compose([
    transforms.Resize((224, 224)),          # 크기만 조정
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])
```

## 2. 표 데이터 파트

### 피처 엔지니어링

**8개 피처:**

1. **is_mixed** (믹스견 여부)
   ```python
   is_mixed = 1 if '믹스' in kindCd or kindCd == 114 else 0
   ```

2. **age_years** (나이)
   ```python
   # 출생년도에서 계산
   age = 2026 - birth_year
   # 60일 미만은 0
   ```

3. **sex_neutered** (성별 + 중성화)
   ```python
   # 0: 수컷 중성화
   # 1: 수컷 미중성화
   # 2: 암컷 중성화
   # 3: 암컷 미중성화
   # 4: 알 수 없음
   ```

4. **weight_kg** (추정 체중)
   ```python
   # 품종명으로 추정
   # 소형견: 5kg (치와와, 말티즈)
   # 중형견: 15kg (비글, 코카)
   # 대형견: 30kg (리트리버, 허스키)
   ```

5. **health_score** (건강 점수)
   ```python
   # specialMark에서 추출
   score = 0
   score += count(['순함', '귀여움', '건강'])  # +1
   score -= count(['겁많음', '피부병', '공격'])  # -1
   ```

6. **has_attack** (공격성)
   ```python
   has_attack = 1 if '공격' in specialMark or '물' in specialMark else 0
   ```

7. **care_encoded** (보호소 인코딩)
   ```python
   # LabelEncoder로 변환
   # 각 보호소마다 고유 번호
   ```

8. **org_encoded** (지역 인코딩)
   ```python
   # LabelEncoder로 변환
   # 각 지역마다 고유 번호
   ```

### 정규화

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_table_scaled = scaler.fit_transform(X_table)

# 평균 0, 표준편차 1로 변환
```

## 3. 멀티모달 결합

### 피처 Concatenation

```python
# CNN 피처: (N, 1280)
# 표 피처: (N, 8)

X_combined = np.concatenate([X_cnn, X_table_scaled], axis=1)
# 결과: (N, 1288)
```

**차원 구성:**
- CNN: 1280차원 (99.4%)
- 표 데이터: 8차원 (0.6%)

## 4. XGBoost 분류기

### 하이퍼파라미터

```python
XGBClassifier(
    objective='multi:softmax',     # 다중 분류
    num_class=3,                   # 3개 클래스
    max_depth=4,                   # 과적합 방지
    learning_rate=0.05,            # 낮은 학습률
    n_estimators=100,              # 트리 개수
    subsample=0.7,                 # 샘플 비율
    colsample_bytree=0.7,          # 피처 비율
    reg_alpha=0.1,                 # L1 정규화
    reg_lambda=1.0,                # L2 정규화
    random_state=42
)
```

**선택 이유:**
- 작은 데이터셋 (277건)에 적합
- 트리 기반 → 피처 중요도 분석 가능
- 정규화로 과적합 방지

### Class Weighting

```python
# 클래스 불균형 처리
class_counts = [생존: 157, 자연사: 57, 안락사: 63]
class_weights = 1.0 / class_counts

# XGBoost의 sample_weight으로 전달
```

## 5. 학습 과정

### 데이터 분할

```
전체 277건
├─ Train: 193건 (70%)
├─ Val:    42건 (15%)
└─ Test:   42건 (15%)

Stratified Split (클래스 비율 유지)
```

### 학습 루프

```python
for epoch in range(100):
    # 1. 훈련
    model.fit(X_train, y_train)
    
    # 2. 검증
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    # 3. Early Stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_model()
```

## 6. 최종 아키텍처 요약

### 입력
- 이미지: 224×224×3 RGB
- 표 데이터: 8개 피처

### 중간 레이어
- CNN 피처: 1280차원
- 표 피처: 8차원 (정규화)
- 결합: 1288차원

### 출력
- 3개 클래스 확률
- Softmax로 정규화

### 성능
- 정확도: 83.3%
- is_mixed 중요도: 0.31% (65위)
- CNN 기여도: 99%

---

## 개선 가능한 부분

1. **앙상블**: EfficientNet + ResNet + MobileNet
2. **Attention**: 이미지의 어느 부분이 중요한지 시각화
3. **더 많은 피처**: 보호소 규모, 봉사자 수 등
4. **데이터 증강**: 277건 → 1000건+

---

## 참고 문헌

- EfficientNet: https://arxiv.org/abs/1905.11946
- XGBoost: https://arxiv.org/abs/1603.02754
- Transfer Learning: https://arxiv.org/abs/1411.1792
