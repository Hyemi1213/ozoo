# 학습 과정

## 전체 타임라인

```
Phase 1: XGBoost 베이스라인  (Week 1-2)
    ↓
Phase 2: CNN 단독 모델       (Week 3-4)
    ↓
Phase 3: 멀티모달 최종       (Week 5-6)
```

---

## Phase 1: XGBoost 베이스라인

### 목표
표 데이터만으로 생존 예측 모델 구축

### 데이터
- 3,302건 (경기 + 부산)
- 20개 피처 (shelter_total_count, is_mixed, weight_kg 등)

### 결과
```
정확도: 75.1%
상위 피처:
  1. is_mixed: 0.25 (순종견 편향!)
  2. shelter_total_count: 0.13
  3. has_health_issue: 0.11
```

### 문제 발견
- ❌ **순종견 편향 심각**: is_mixed가 1위
- ❌ 순종견 → 무조건 생존 예측
- ❌ 믹스견 → 안락사 과대평가

**→ 이미지 데이터 필요!**

---

## Phase 2: CNN 단독 모델

### 2-1. 이미지 다운로드

**데이터 수집:**
```
API: 농림축산검역본부 유기동물 공공API
기간: 2025년 10-12월
지역: 경기, 부산
```

**다운로드 결과:**
```
전체 이미지 URL: 9,000개
성공: 800개
실패: 8,200개 (링크 깨짐, 타임아웃)
유니크 개체: 361마리
```

### 2-2. CNN 학습 (부산 데이터만)

**데이터:**
- 이미지: 121건
- 클래스: 생존 85, 자연사 25, 안락사 11

**모델:**
- EfficientNet-B0 (ImageNet 사전학습)
- Batch size: 16
- Epochs: 25

**결과:**
```
Train Acc: 97.6%
Val Acc:   77.8%
Test Acc:  63.2%

→ 과적합 발생!
```

**문제:**
- ❌ 데이터 부족 (121건)
- ❌ 과적합 (Train 97%, Test 63%)

### 2-3. CNN 학습 (전체 데이터)

**데이터:**
- 이미지: 277건
- 클래스: 생존 157, 자연사 57, 안락사 63

**개선 사항:**
```python
# 1. 강력한 Augmentation
RandomRotation(30)  # 20 → 30
ColorJitter(0.4)    # 0.3 → 0.4
RandomPerspective   # 추가

# 2. 정규화
reg_alpha=0.1
reg_lambda=1.0

# 3. Early Stopping
patience=10
```

**학습 곡선:**
```
Epoch  Train_Acc  Val_Acc
1      41.7%      72.2%
4      94.1%      77.8%  ← Best
8      98.8%      77.8%
14     94.1%      66.7%  ← Early Stop
```

**최종 결과:**
```
Best Val Acc: 77.8%
Test Acc:     63.2%

→ 여전히 과적합
→ 데이터 더 필요!
```

---

## Phase 3: 멀티모달 최종

### 3-1. 피처 생성

**표 데이터 피처 엔지니어링:**
```python
# 1. is_mixed
is_mixed = 1 if '믹스' in kindCd else 0

# 2. age_years
age = 2026 - birth_year

# 3. sex_neutered
# M+Y:0, M+N:1, F+Y:2, F+N:3, Unknown:4

# 4. weight_kg
# 품종명으로 추정: 소형 5kg, 중형 15kg, 대형 30kg

# 5. health_score
score = count(긍정어) - count(부정어)

# 6. has_attack
has_attack = 1 if '공격' in specialMark else 0

# 7-8. care_encoded, org_encoded
# LabelEncoder
```

### 3-2. CNN 피처 추출

```python
# 사전 학습된 CNN에서 피처 추출
feature_extractor = FeatureExtractor(cnn_model)

cnn_features = []
for image in images:
    features = feature_extractor(image)  # (1280,)
    cnn_features.append(features)

X_cnn = np.array(cnn_features)  # (277, 1280)
```

### 3-3. 피처 결합

```python
# 표 데이터 정규화
scaler = StandardScaler()
X_table_scaled = scaler.fit_transform(X_table)  # (277, 8)

# CNN + 표 데이터
X_combined = np.concatenate([X_cnn, X_table_scaled], axis=1)
# (277, 1288)
```

### 3-4. XGBoost 학습

**데이터 분할:**
```
Train: 193건 (70%)
Val:    42건 (15%)
Test:   42건 (15%)
```

**하이퍼파라미터 튜닝:**
```python
# 시도 1: 기본값
max_depth=6, lr=0.1, n_est=200
→ Val Acc: 76.2% (과적합)

# 시도 2: 정규화 강화
max_depth=4, lr=0.05, n_est=100
→ Val Acc: 81.0% ✓

# 시도 3: 더 강한 정규화
max_depth=3, lr=0.03, n_est=150
→ Val Acc: 78.6% (언더피팅)

# 최종: 시도 2 선택
```

**학습 과정:**
```
[0]    train: 0.967, val: 0.999
[20]   train: 0.456, val: 0.857
[40]   train: 0.246, val: 0.805
[60]   train: 0.146, val: 0.794
[80]   train: 0.097, val: 0.797
[99]   train: 0.070, val: 0.804

→ 안정적 수렴
```

**최종 결과:**
```
Train Acc: 95.3%
Val Acc:   81.0%
Test Acc:  83.3% ✓✓✓

→ 과적합 최소화 성공!
```

---

## 성능 비교

| Phase | 모델 | 데이터 | 정확도 | 비고 |
|-------|------|--------|--------|------|
| 1 | XGBoost | 표 (3,302) | 75.1% | 순종견 편향 |
| 2-1 | CNN | 이미지 (121) | 63.2% | 과적합 |
| 2-2 | CNN | 이미지 (277) | 63.2% | 여전히 부족 |
| 3 | **멀티모달** | **이미지+표 (277)** | **83.3%** | **최종** ✓ |

---

## 학습 교훈

### ✅ 성공 요인

1. **멀티모달 접근**
   - CNN (이미지) + 표 데이터
   - 서로의 약점 보완

2. **강력한 정규화**
   - Data Augmentation
   - L1/L2 정규화
   - Early Stopping

3. **적절한 모델 선택**
   - EfficientNet: 경량 + 고성능
   - XGBoost: 작은 데이터에 적합

### ⚠️ 실패 사례

1. **데이터 부족**
   - 121건: 과적합 심각
   - 277건: 개선되었지만 여전히 적음
   - 해결: 멀티모달로 극복

2. **하이퍼파라미터**
   - 처음: max_depth=6 → 과적합
   - 수정: max_depth=4 → 성공

3. **초기 CNN**
   - Augmentation 부족
   - 정규화 부족
   - 해결: 강화된 설정

---

## 개선 가능한 부분

### 1. 더 많은 데이터
```
현재: 277건
목표: 1,000건+

방법:
- 더 많은 지역 (서울, 인천 등)
- 더 긴 기간 (6개월 → 1년)
- API 안정화
```

### 2. 앙상블
```python
# 3개 모델 투표
model1 = EfficientNet-B0
model2 = ResNet50
model3 = MobileNetV2

final_pred = vote([model1, model2, model3])

예상: 83.3% → 85%+
```

### 3. 피처 추가
```
현재: 8개 피처
추가 가능:
- shelter_size (보호소 규모)
- volunteer_count (봉사자 수)
- regional_income (지역 소득)
- adoption_rate (입양률)
```

### 4. Attention 메커니즘
```python
# 이미지의 어느 부분이 중요한지
attention_map = model.attention(image)

시각화:
- 눈: 건강 상태
- 털: 관리 상태
- 자세: 성격
```

---

## 시간 소요

```
Phase 1 (XGBoost):     2주
Phase 2 (CNN):         2주
Phase 3 (멀티모달):    2주
---------------------------------
총:                    6주
```

**실제 작업 시간:**
```
데이터 수집/전처리: 20시간
모델 개발/실험:     40시간
분석/문서화:        15시간
---------------------------------
총:                 75시간
```

---

## 핵심 포인트

1. **멀티모달이 핵심**
   - CNN 단독: 63.2%
   - 멀티모달: 83.3%
   - 차이: +20.1%p!

2. **순종견 편향 제거**
   - XGBoost: 1위 (25%)
   - 멀티모달: 65위 (0.31%)
   - 감소: 98.7%!

3. **이미지의 중요성**
   - CNN 기여도: 99%
   - 외관이 생존 예측의 핵심

4. **작은 데이터도 가능**
   - 277건으로 83.3% 달성
   - 전이 학습 + 정규화가 핵심

---

## 다음 단계

1. **프로덕션 배포**
   - FastAPI로 API 서버
   - Docker 컨테이너화
   - AWS/GCP 배포

2. **실시간 예측**
   - 보호소에서 사진 촬영 즉시 예측
   - 고위험 개체 자동 알림

3. **대시보드**
   - Streamlit/Gradio
   - 사진 업로드 → 즉시 예측
   - 설명 가능한 AI (SHAP, Attention)
