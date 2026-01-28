# RescueAI - 유기동물 골든타임 확보 시스템

> **순종견 편향 98.7% 감소, 83.3% 정확도 달성**
> CNN 이미지 분석 + 표 데이터를 결합한 멀티모달 AI

## 프로젝트 개요

유기동물 보호소에서 입소 개체의 **위험도를 조기에 탐지**하여 골든타임을 확보하는 AI 시스템입니다.

### 핵심 성과

| 항목 | 결과 |
|------|------|
| 정확도 | 83.3% (XGBoost 75.1% 대비 +8.2%p) |
| 순종견 편향 감소 | 98.7% (1위 → 65위) |
| 아키텍처 | CNN + 표 데이터 멀티모달 |
| 이미지 기여도 | 99% |

### 문제 정의

기존 XGBoost 모델의 순종견 편향:
- `is_mixed` 피처가 1위 (25% 중요도)
- 순종견 = 무조건 생존 예측
- 믹스견 = 안락사 과대평가

### 해결 방안

멀티모달 AI로 편향 제거:
- CNN: 이미지에서 외관, 건강, 표정 분석
- 표 데이터: 나이, 체중, 보호소 정보
- 결합: XGBoost로 최종 예측

## 아키텍처

```
이미지 → EfficientNet-B0 → 1280차원
                              ↓
표 데이터 → 피처 생성 → 8차원 → 결합 (1288차원) → XGBoost → 예측
```

## 성능 비교

| 모델 | 정확도 | is_mixed 순위 | 순종견 편향 |
|------|--------|---------------|-------------|
| XGBoost | 75.1% | 1위 (25%) | 매우 심각 |
| CNN | 63.2% | - | - |
| **멀티모달** | **83.3%** | **65위 (0.31%)** | **98.7% 감소** |

## Quick Start

### 설치

```bash
pip install -r requirements.txt
```

### Web Demo 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속하면 데모를 확인할 수 있습니다.

## 프로젝트 구조

```
ozoo/
├── app.py                 # Streamlit 웹 데모
├── requirements.txt       # 패키지 목록
├── README.md
├── .gitignore
│
├── docs/                  # 기술 문서
│   ├── architecture.md
│   ├── training_process.md
│   ├── performance_comparison.md
│   └── bias_analysis.md
│
├── notebooks/             # 실험 노트북
│   ├── 머신러닝모델.ipynb
│   └── 딥러닝모델.ipynb
│
└── data/
    └── images/            # 보호소 이미지 데이터 (800장)
```

## 기술 문서

- [모델 아키텍처](docs/architecture.md)
- [학습 과정](docs/training_process.md)
- [성능 분석](docs/performance_comparison.md)
- [편향 분석](docs/bias_analysis.md)

## 기술 스택

- **Deep Learning**: PyTorch, EfficientNet-B0
- **Machine Learning**: XGBoost, scikit-learn
- **Web Demo**: Streamlit
- **데이터 처리**: pandas, numpy, Pillow
- **시각화**: matplotlib, seaborn


## 라이센스

MIT License
