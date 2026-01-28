import streamlit as st
import numpy as np
import os
import random
from PIL import Image, ImageOps
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RescueAI - 유기동물 골든타임 확보",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }
    .card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .card-orange {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
    }
    .card-red {
        background: linear-gradient(135deg, #e44d26 0%, #f16529 100%);
    }
    .card-blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    .result-survive {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .result-natural {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
    .result-euthanasia {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "1769494106663-shelter_images")
CLASS_NAMES = ["생존 (입양/반환)", "자연사", "안락사"]
CLASS_COLORS = ["#28a745", "#ffc107", "#dc3545"]
CLASS_SHORT = ["생존", "자연사", "안락사"]

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    font=dict(family="Malgun Gothic, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)

# Model loaded flag
MODEL_AVAILABLE = False
model_bundle = None

# Try loading model if available
MODEL_PKL = os.path.join(os.path.dirname(__file__), "multimodal_model.pkl")
CNN_PTH = os.path.join(os.path.dirname(__file__), "best_cnn_model.pth")

try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
    import torch.nn as nn
    import pickle
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FeatureExtractor(nn.Module if TORCH_AVAILABLE else object):
    """CNN feature extractor that strips the classification head."""
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


@st.cache_resource
def load_model():
    """Load the saved multimodal model bundle if it exists."""
    global MODEL_AVAILABLE
    if not TORCH_AVAILABLE:
        return None

    if os.path.exists(MODEL_PKL):
        with open(MODEL_PKL, "rb") as f:
            bundle = pickle.load(f)
        MODEL_AVAILABLE = True
        return bundle

    if os.path.exists(CNN_PTH):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn = models.efficientnet_b0(weights=None)
        cnn.classifier[1] = nn.Linear(cnn.classifier[1].in_features, 3)
        cnn.load_state_dict(torch.load(CNN_PTH, map_location=device))
        cnn.eval()
        fe = FeatureExtractor(cnn).to(device)
        fe.eval()
        return {"feature_extractor": fe, "device": device}

    return None


def get_sample_images(n=6):
    """Return a random sample of images from the shelter images folder."""
    if not os.path.isdir(IMAGE_FOLDER):
        return []
    all_imgs = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(".jpg")]
    if not all_imgs:
        return []
    selected = random.sample(all_imgs, min(n, len(all_imgs)))
    return [os.path.join(IMAGE_FOLDER, f) for f in selected]


def simulate_prediction(image, tabular_features):
    """Generate a realistic demo prediction when no trained model is available."""
    probs = np.array([0.57, 0.14, 0.29])

    age = tabular_features.get("age_years", 3)
    health = tabular_features.get("health_score", 0)
    has_attack = tabular_features.get("has_attack", 0)
    weight = tabular_features.get("weight_kg", 10)

    if age <= 2:
        probs[0] += 0.15
        probs[2] -= 0.10
    elif age >= 8:
        probs[1] += 0.10
        probs[0] -= 0.08

    if health > 0:
        probs[0] += 0.10
        probs[1] -= 0.05
    elif health < 0:
        probs[1] += 0.08
        probs[0] -= 0.05

    if has_attack:
        probs[2] += 0.15
        probs[0] -= 0.10

    if weight < 3 or weight > 35:
        probs[1] += 0.05

    noise = np.random.normal(0, 0.03, 3)
    probs += noise
    probs = np.clip(probs, 0.01, 1.0)
    probs /= probs.sum()

    return probs


def extract_features_from_image(image_pil, bundle):
    """Extract CNN features from an image using the loaded model."""
    if bundle is None or not TORCH_AVAILABLE:
        return None

    device = bundle.get("device", torch.device("cpu"))
    fe = bundle.get("feature_extractor")
    if fe is None:
        return None

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        features = fe(img_tensor).cpu().numpy().flatten()
    return features


# ──────────────────────────────────────────────
# Plotly helper: confusion matrix heatmap
# ──────────────────────────────────────────────
def plotly_confusion_matrix(cm, title):
    labels = CLASS_SHORT
    text = [[str(val) for val in row] for row in cm]
    fig = go.Figure(data=go.Heatmap(
        z=cm[::-1],
        x=labels,
        y=labels[::-1],
        text=text[::-1],
        texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        colorscale="Blues",
        showscale=False,
        hovertemplate="실제: %{y}<br>예측: %{x}<br>건수: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="예측",
        yaxis_title="실제",
        height=320,
        **PLOTLY_LAYOUT,
    )
    return fig


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("RescueAI")
    st.caption("유기동물 골든타임 확보 시스템")
    st.divider()

    page = st.radio(
        "페이지",
        ["프로젝트 개요", "모델 아키텍처", "성능 비교", "편향 분석", "예측 데모"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("SeSAC 화이팅")
   


# ──────────────────────────────────────────────
# Page 1: Project Overview
# ──────────────────────────────────────────────
if page == "프로젝트 개요":
    st.title("RescueAI")
    st.subheader("유기동물 골든타임 확보를 위한 멀티모달 AI 시스템")
    st.markdown("CNN 이미지 분석 + 표 데이터를 결합하여 유기동물의 **위험도를 조기 탐지**하고 골든타임을 확보합니다.")

    st.divider()

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="metric-card card-green"><h2>83.3%</h2><p>최종 정확도</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="metric-card"><h2>98.7%</h2><p>순종견 편향 감소</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="metric-card card-orange"><h2>+8.2%p</h2><p>XGBoost 대비 향상</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="metric-card card-blue"><h2>99%</h2><p>이미지 기여도</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Problem & Solution
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 문제 정의")
        st.error(
            "기존 XGBoost 모델의 **순종견 편향**\n\n"
            "- `is_mixed` 피처가 중요도 1위 (25%)\n"
            "- 순종견 → 무조건 생존 예측\n"
            "- 믹스견 → 안락사 과대평가"
        )
    with col_b:
        st.markdown("### 해결 방안")
        st.success(
            "**멀티모달 AI**로 편향 제거\n\n"
            "- CNN: 이미지에서 외관/건강/표정 분석\n"
            "- 표 데이터: 나이, 체중, 보호소 정보\n"
            "- XGBoost로 최종 예측 (1288차원)"
        )

    st.divider()

    # Dataset info
    st.markdown("### 데이터셋")
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.metric("전체 데이터", "3,331건")
    with dc2:
        st.metric("이미지 매칭", "277건")
    with dc3:
        st.metric("보호소 이미지", "800장")

    # Class distribution chart (plotly)
    st.markdown("#### 클래스 분포")
    labels_dist = ["생존", "자연사", "안락사"]
    sizes_dist = [157, 57, 63]
    colors_dist = ["#28a745", "#ffc107", "#dc3545"]

    fig_dist = go.Figure(go.Bar(
        y=labels_dist,
        x=sizes_dist,
        orientation="h",
        marker_color=colors_dist,
        text=[f"{v}건" for v in sizes_dist],
        textposition="outside",
        textfont=dict(size=13),
        hovertemplate="%{y}: %{x}건<extra></extra>",
    ))
    fig_dist.update_layout(
        title="이미지 매칭 데이터 클래스 분포 (277건)",
        xaxis_title="샘플 수",
        xaxis=dict(range=[0, max(sizes_dist) * 1.25]),
        height=280,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Sample images
    if os.path.isdir(IMAGE_FOLDER):
        st.markdown("#### 보호소 이미지 샘플")
        sample_imgs = get_sample_images(6)
        if sample_imgs:
            img_cols = st.columns(6)
            for i, path in enumerate(sample_imgs):
                with img_cols[i]:
                    img = Image.open(path)
                    st.image(img, width="stretch",
                             caption=os.path.basename(path).split("_")[0])


# ──────────────────────────────────────────────
# Page 2: Model Architecture
# ──────────────────────────────────────────────
elif page == "모델 아키텍처":
    st.title("모델 아키텍처")
    st.subheader("CNN + 표 데이터 멀티모달 결합")

    st.divider()

    st.markdown("### 전체 파이프라인")
    st.code(
        """
 ┌─────────────────────┐           ┌──────────────────────┐
 │    이미지 입력        │           │    표 데이터 입력      │
 │   (224 x 224 x 3)   │           │     (8 features)     │
 └──────────┬──────────┘           └──────────┬───────────┘
            │                                 │
            ▼                                 ▼
 ┌─────────────────────┐           ┌──────────────────────┐
 │  EfficientNet-B0    │           │   Feature Eng.       │
 │  (ImageNet 사전학습)  │           │   is_mixed, age,     │
 │                     │           │   weight, health ...  │
 │  → Conv Blocks      │           └──────────┬───────────┘
 │  → MBConv           │                      │
 │  → AvgPool          │                      ▼
 └──────────┬──────────┘           ┌──────────────────────┐
            │                      │   StandardScaler     │
            ▼                      │    (정규화)            │
 ┌─────────────────────┐           └──────────┬───────────┘
 │  CNN Features       │                      │
 │   (1280 dims)       │                      │
 └──────────┬──────────┘                      │
            │                                 │
            └───────────────┬─────────────────┘
                            │
                            ▼
                 ┌──────────────────┐
                 │   Concatenate    │
                 │   (1288 dims)    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    XGBoost       │
                 │   Classifier     │
                 │                  │
                 │  max_depth: 4    │
                 │  lr: 0.05        │
                 │  n_est: 100      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   3-class 출력    │
                 │                  │
                 │  0: 생존          │
                 │  1: 자연사        │
                 │  2: 안락사        │
                 └──────────────────┘
        """,
        language=None,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### CNN 파트 (이미지)")
        st.markdown("""
| 항목 | 설정 |
|------|------|
| 모델 | EfficientNet-B0 |
| 파라미터 | 5.3M |
| 사전학습 | ImageNet |
| 출력 | 1280차원 피처 벡터 |
| 입력 크기 | 224 x 224 x 3 |
        """)
        st.markdown("**Data Augmentation (학습 시)**")
        st.markdown("""
- `RandomCrop(224)` - 랜덤 크롭
- `RandomHorizontalFlip(0.5)` - 좌우 반전
- `RandomRotation(20)` - 회전
- `ColorJitter(0.3)` - 색상 변경
- `RandomAffine` - 아핀 변환
        """)

    with col2:
        st.markdown("### 표 데이터 파트 (8개 피처)")
        st.markdown("""
| 피처 | 설명 | 타입 |
|------|------|------|
| `is_mixed` | 믹스견 여부 | Binary |
| `age_years` | 나이 (년) | Numeric |
| `sex_neutered` | 성별+중성화 | Categorical |
| `weight_kg` | 추정 체중 | Numeric |
| `health_score` | 건강 점수 | Numeric |
| `has_attack` | 공격성 여부 | Binary |
| `care_encoded` | 보호소 코드 | Encoded |
| `org_encoded` | 지역 코드 | Encoded |
        """)

    st.divider()

    st.markdown("### XGBoost 최종 분류기")
    hp1, hp2 = st.columns(2)
    with hp1:
        st.markdown("""
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `objective` | multi:softmax | 다중 분류 |
| `num_class` | 3 | 클래스 수 |
| `max_depth` | 4 | 과적합 방지 |
| `learning_rate` | 0.05 | 낮은 학습률 |
| `n_estimators` | 100 | 트리 개수 |
        """)
    with hp2:
        st.markdown("""
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `subsample` | 0.7 | 샘플 비율 |
| `colsample_bytree` | 0.7 | 피처 비율 |
| `reg_alpha` | 0.1 | L1 정규화 |
| `reg_lambda` | 1.0 | L2 정규화 |
| `random_state` | 42 | 재현성 |
        """)

    st.info(
        "**왜 XGBoost?** 작은 데이터셋(277건)에서 트리 기반 모델이 "
        "신경망보다 안정적이며, 피처 중요도 분석이 가능합니다."
    )


# ──────────────────────────────────────────────
# Page 3: Performance Comparison
# ──────────────────────────────────────────────
elif page == "성능 비교":
    st.title("성능 비교 분석")
    st.subheader("XGBoost vs CNN vs 멀티모달")

    st.divider()

    # Accuracy comparison (plotly)
    st.markdown("### 전체 정확도")
    model_names = ["XGBoost (표 데이터)", "CNN (이미지)", "멀티모달 (최종)"]
    train_accs = [85.2, 97.6, 95.3]
    val_accs = [78.3, 77.8, 81.0]
    test_accs = [75.1, 63.2, 83.3]

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(name="Train", x=model_names, y=train_accs,
                             marker_color="#6c757d", opacity=0.7,
                             text=[f"{v:.1f}%" for v in train_accs], textposition="outside"))
    fig_acc.add_trace(go.Bar(name="Validation", x=model_names, y=val_accs,
                             marker_color="#007bff", opacity=0.85,
                             text=[f"{v:.1f}%" for v in val_accs], textposition="outside"))
    fig_acc.add_trace(go.Bar(name="Test", x=model_names, y=test_accs,
                             marker_color="#28a745", opacity=0.9,
                             text=[f"{v:.1f}%" for v in test_accs], textposition="outside"))
    fig_acc.add_hline(y=83.3, line_dash="dash", line_color="#28a745", opacity=0.35,
                      annotation_text="멀티모달 83.3%", annotation_position="top left")
    fig_acc.update_layout(
        barmode="group",
        title="모델별 정확도 비교",
        yaxis_title="정확도 (%)",
        yaxis=dict(range=[50, 108]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=450,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.divider()

    # Per-class metrics
    st.markdown("### 클래스별 성능 (Test Set)")
    tab1, tab2, tab3 = st.tabs(["생존", "자연사", "안락사"])

    with tab1:
        st.markdown("#### 생존 (Survival)")
        st.table({
            "모델": ["XGBoost", "CNN", "멀티모달"],
            "Precision": [0.78, 0.77, 0.89],
            "Recall": [0.85, 0.77, 0.89],
            "F1-Score": [0.81, 0.77, 0.89],
        })
    with tab2:
        st.markdown("#### 자연사 (Natural Death)")
        st.table({
            "모델": ["XGBoost", "CNN", "멀티모달"],
            "Precision": [0.67, 0.50, 0.71],
            "Recall": [0.50, 0.60, 0.71],
            "F1-Score": [0.57, 0.55, 0.71],
        })
    with tab3:
        st.markdown("#### 안락사 (Euthanasia)")
        st.table({
            "모델": ["XGBoost", "CNN", "멀티모달"],
            "Precision": [0.76, 0.00, 0.82],
            "Recall": [0.79, 0.00, 0.82],
            "F1-Score": [0.77, 0.00, 0.82],
        })

    st.divider()

    # F1-Score comparison (plotly)
    st.markdown("### 클래스별 F1-Score 비교")
    classes = ["생존", "자연사", "안락사"]
    fig_f1 = go.Figure()
    fig_f1.add_trace(go.Bar(name="XGBoost", x=classes, y=[0.81, 0.57, 0.77],
                            marker_color="#6c757d", opacity=0.85))
    fig_f1.add_trace(go.Bar(name="CNN", x=classes, y=[0.77, 0.55, 0.00],
                            marker_color="#007bff", opacity=0.85))
    fig_f1.add_trace(go.Bar(name="멀티모달", x=classes, y=[0.89, 0.71, 0.82],
                            marker_color="#28a745", opacity=0.9))
    fig_f1.update_layout(
        barmode="group",
        title="클래스별 F1-Score",
        yaxis_title="F1-Score",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=400,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    st.divider()

    # Confusion Matrices (plotly)
    st.markdown("### 혼동 행렬 (Confusion Matrix)")
    cm_col1, cm_col2, cm_col3 = st.columns(3)

    with cm_col1:
        st.plotly_chart(
            plotly_confusion_matrix(
                np.array([[17, 1, 2], [2, 4, 2], [2, 1, 11]]), "XGBoost"
            ),
            use_container_width=True,
        )
    with cm_col2:
        st.plotly_chart(
            plotly_confusion_matrix(
                np.array([[10, 3, 0], [1, 3, 1], [1, 0, 0]]), "CNN"
            ),
            use_container_width=True,
        )
    with cm_col3:
        st.plotly_chart(
            plotly_confusion_matrix(
                np.array([[16, 1, 1], [1, 5, 1], [2, 1, 14]]), "멀티모달 (최종)"
            ),
            use_container_width=True,
        )

    st.divider()

    # ROC-AUC
    st.markdown("### ROC-AUC 분석")
    st.table({
        "모델": ["XGBoost", "CNN", "멀티모달"],
        "생존": [0.88, 0.85, 0.92],
        "자연사": [0.75, 0.70, 0.85],
        "안락사": [0.82, 0.65, 0.89],
        "Macro Avg": [0.82, 0.73, 0.89],
    })

    # Summary evaluation (plotly)
    st.markdown("### 종합 평가")
    categories = ["정확도", "편향 제거", "일반화", "해석성", "속도"]

    fig_summary = go.Figure()
    fig_summary.add_trace(go.Bar(name="XGBoost", x=categories,
                                 y=[7.5, 3.0, 7.0, 9.0, 10.0],
                                 marker_color="#6c757d", opacity=0.85))
    fig_summary.add_trace(go.Bar(name="CNN", x=categories,
                                 y=[6.3, 8.0, 5.0, 5.0, 6.0],
                                 marker_color="#007bff", opacity=0.85))
    fig_summary.add_trace(go.Bar(name="멀티모달", x=categories,
                                 y=[8.3, 9.9, 8.0, 7.0, 6.0],
                                 marker_color="#28a745", opacity=0.9))
    fig_summary.update_layout(
        barmode="group",
        title="종합 평가 (10점 만점)",
        yaxis_title="점수",
        yaxis=dict(range=[0, 11]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_summary, use_container_width=True)


# ──────────────────────────────────────────────
# Page 4: Bias Analysis
# ──────────────────────────────────────────────
elif page == "편향 분석":
    st.title("순종견 편향 분석")
    st.subheader("AI 공정성을 위한 편향 측정 및 제거")

    st.divider()

    st.markdown("### 순종견 편향이란?")
    st.warning(
        "모델이 순종견(`is_mixed=0`)이라는 정보만으로 "
        "생존 확률을 **과도하게** 높게 예측하는 현상\n\n"
        "- 순종견 + 나이 많음 + 건강 나쁨 → **생존 예측** (편향)\n"
        "- 믹스견 + 어림 + 건강 좋음 → **안락사 예측** (편향)"
    )

    st.divider()

    # Feature importance comparison (plotly)
    st.markdown("### 피처 중요도 변화")

    fi_col1, fi_col2 = st.columns(2)

    with fi_col1:
        st.markdown("#### XGBoost (표 데이터만)")
        features_xgb = ["weight_kg", "age_years", "has_health", "shelter_total", "is_mixed"]
        importances_xgb = [0.08, 0.09, 0.11, 0.13, 0.25]
        colors_fi1 = ["#6c757d"] * 4 + ["#dc3545"]

        fig_fi1 = go.Figure(go.Bar(
            y=features_xgb, x=importances_xgb, orientation="h",
            marker_color=colors_fi1,
            text=[f"{v:.0%}" for v in importances_xgb],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        fig_fi1.update_layout(
            title="XGBoost 피처 중요도",
            xaxis_title="중요도",
            xaxis=dict(range=[0, 0.32]),
            height=300,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_fi1, use_container_width=True)
        st.error("**is_mixed가 1위 (25%)** - 순종견 편향 심각!")

    with fi_col2:
        st.markdown("#### 멀티모달")
        features_multi = ["CNN_853", "CNN_708", "CNN_374", "CNN_1022", "CNN_567"]
        importances_multi = [0.0068, 0.0076, 0.0083, 0.0097, 0.0104]

        fig_fi2 = go.Figure(go.Bar(
            y=features_multi, x=importances_multi, orientation="h",
            marker_color="#28a745", opacity=0.85,
            text=[f"{v:.2%}" for v in importances_multi],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        ))
        fig_fi2.update_layout(
            title="멀티모달 Top-5 피처 중요도",
            xaxis_title="중요도",
            xaxis=dict(range=[0, 0.014]),
            height=300,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_fi2, use_container_width=True)
        st.success("**is_mixed가 65위 (0.31%)** - CNN 피처가 지배적!")

    st.divider()

    # Bias reduction metrics
    st.markdown("### 편향 감소 지표")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            '<div class="metric-card card-green"><h2>98.7%</h2>'
            '<p>is_mixed 중요도 감소<br>1위 → 65위</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '<div class="metric-card card-blue"><h2>53%</h2>'
            '<p>순종-믹스 정확도 차이 감소<br>25%p → 11.7%p</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            '<div class="metric-card card-orange"><h2>0.07</h2>'
            '<p>Demographic Parity<br>0.27 → 0.07</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Breed accuracy comparison (plotly)
    st.markdown("### 순종견 vs 믹스견 정확도")

    breed_labels = ["XGBoost<br>순종견", "XGBoost<br>믹스견", "멀티모달<br>순종견", "멀티모달<br>믹스견"]
    breed_accs = [85, 60, 91.7, 80.0]
    breed_colors = ["#6c757d", "#adb5bd", "#28a745", "#7dcea0"]

    fig_bias = go.Figure(go.Bar(
        x=breed_labels, y=breed_accs,
        marker_color=breed_colors,
        text=[f"{v:.1f}%" for v in breed_accs],
        textposition="outside",
        textfont=dict(size=13, color="#333"),
        width=0.5,
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    # Gap annotations
    fig_bias.add_annotation(
        x=0.5, y=72.5, text="<b>25%p</b>", showarrow=False,
        font=dict(size=14, color="#dc3545"),
        bgcolor="white", bordercolor="#dc3545", borderwidth=2, borderpad=4,
    )
    fig_bias.add_shape(type="line", x0=0.5, x1=0.5, y0=60, y1=85,
                       line=dict(color="#dc3545", width=2.5, dash="dot"))
    fig_bias.add_annotation(
        x=2.5, y=85.8, text="<b>11.7%p</b>", showarrow=False,
        font=dict(size=14, color="#28a745"),
        bgcolor="white", bordercolor="#28a745", borderwidth=2, borderpad=4,
    )
    fig_bias.add_shape(type="line", x0=2.5, x1=2.5, y0=80, y1=91.7,
                       line=dict(color="#28a745", width=2.5, dash="dot"))
    fig_bias.update_layout(
        title="순종견 vs 믹스견 정확도 편향 비교",
        yaxis_title="정확도 (%)",
        yaxis=dict(range=[40, 100]),
        height=420,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_bias, use_container_width=True)

    st.divider()

    # Fairness metrics
    st.markdown("### 공정성 지표 비교")
    st.table({
        "지표": ["Demographic Parity", "Equal Opportunity", "편향도 (정확도 차이)"],
        "XGBoost": ["0.27 (심각)", "0.25 (심각)", "25%p"],
        "멀티모달": ["0.07 (양호)", "0.08 (양호)", "11.7%p"],
        "개선율": ["74%", "68%", "53%"],
    })

    st.divider()

    # Case studies
    st.markdown("### 사례 분석")
    case1, case2 = st.columns(2)
    with case1:
        st.markdown("#### 나이든 순종견")
        st.markdown("""
| | XGBoost | 멀티모달 |
|---|---------|----------|
| **예측** | 생존 | 생존 |
| **이유** | is_mixed=0 | 이미지 분석 |
| **실제** | 생존 (입양) | 생존 (입양) |
        """)
        st.info("멀티모달: is_mixed가 아닌 **이미지 기반** 균형잡힌 판단")
    with case2:
        st.markdown("#### 어린 믹스견")
        st.markdown("""
| | XGBoost | 멀티모달 |
|---|---------|----------|
| **예측** | 안락사 | 생존 |
| **이유** | is_mixed=1 | 귀여워 보임 |
| **실제** | 생존 (입양) | 생존 (입양) |
        """)
        st.success("멀티모달: CNN이 순종견 편향을 **제거**하여 올바른 예측")


# ──────────────────────────────────────────────
# Page 5: Prediction Demo
# ──────────────────────────────────────────────
elif page == "예측 데모":
    st.title("예측 데모")
    st.subheader("이미지 + 표 데이터로 유기동물 위험도 예측")

    bundle = load_model()
    has_model = bundle is not None and MODEL_AVAILABLE

    if not has_model:
        st.info(
            "학습된 모델 파일이 없어 **데모 모드**로 동작합니다. "
            "실제 모델 가중치(`multimodal_model.pkl`)를 프로젝트 루트에 배치하면 "
            "실시간 추론이 가능합니다."
        )

    st.divider()

    col_img, col_form = st.columns([1, 1])

    with col_img:
        st.markdown("### 이미지 입력")
        input_method = st.radio(
            "이미지 입력 방식",
            ["직접 업로드", "샘플 이미지 선택"],
            horizontal=True,
        )

        uploaded_image = None

        if input_method == "직접 업로드":
            uploaded_file = st.file_uploader(
                "유기동물 사진을 업로드하세요",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_file:
                uploaded_image = Image.open(uploaded_file).convert("RGB")
        else:
            sample_paths = get_sample_images(12)
            if sample_paths:
                sample_names = [os.path.basename(p).split("_")[0] for p in sample_paths]
                selected_sample = st.selectbox(
                    "샘플 이미지를 선택하세요",
                    range(len(sample_paths)),
                    format_func=lambda i: f"개체 {sample_names[i]}",
                )
                uploaded_image = Image.open(sample_paths[selected_sample]).convert("RGB")
            else:
                st.warning("샘플 이미지 폴더를 찾을 수 없습니다.")

        if uploaded_image:
            display_img = ImageOps.contain(uploaded_image, (400, 400))
            st.image(display_img, caption="입력 이미지", width="stretch")

    with col_form:
        st.markdown("### 표 데이터 입력")

        is_mixed = st.selectbox("품종", ["순종견", "믹스견"])
        age_years = st.slider("나이 (년)", 0, 20, 3)

        sex_opt = st.selectbox(
            "성별 / 중성화",
            ["수컷 (중성화 O)", "수컷 (중성화 X)", "암컷 (중성화 O)", "암컷 (중성화 X)", "알 수 없음"],
        )
        sex_map = {
            "수컷 (중성화 O)": 0, "수컷 (중성화 X)": 1,
            "암컷 (중성화 O)": 2, "암컷 (중성화 X)": 3,
            "알 수 없음": 4,
        }

        weight_kg = st.slider("추정 체중 (kg)", 1.0, 50.0, 10.0, 0.5)
        health_score = st.slider("건강 점수", -3, 3, 0,
                                 help="양수: 건강/순함, 음수: 질병/공격적")
        has_attack = st.checkbox("공격성 있음")

    st.divider()

    if st.button("예측하기", type="primary", use_container_width=True):
        if uploaded_image is None:
            st.warning("이미지를 먼저 입력해주세요.")
        else:
            tabular = {
                "is_mixed": 1 if is_mixed == "믹스견" else 0,
                "age_years": age_years,
                "sex_neutered": sex_map[sex_opt],
                "weight_kg": weight_kg,
                "health_score": health_score,
                "has_attack": int(has_attack),
                "care_encoded": 0,
                "org_encoded": 0,
            }

            with st.spinner("예측 중..."):
                if has_model:
                    cnn_feats = extract_features_from_image(uploaded_image, bundle)
                    scaler = bundle.get("scaler")
                    xgb_model = bundle.get("model")

                    tab_arr = np.array([[
                        tabular["is_mixed"], tabular["age_years"],
                        tabular["sex_neutered"], tabular["weight_kg"],
                        tabular["health_score"], tabular["has_attack"],
                        tabular["care_encoded"], tabular["org_encoded"],
                    ]])

                    if scaler is not None:
                        tab_arr = scaler.transform(tab_arr)

                    combined = np.concatenate([cnn_feats.reshape(1, -1), tab_arr], axis=1)
                    probs = xgb_model.predict_proba(combined)[0]
                else:
                    probs = simulate_prediction(uploaded_image, tabular)

            pred_class = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_class]
            pred_conf = probs[pred_class] * 100

            st.markdown("### 예측 결과")

            result_styles = ["result-survive", "result-natural", "result-euthanasia"]
            result_icons = ["🟢", "🟡", "🔴"]

            st.markdown(
                f'<div class="result-box {result_styles[pred_class]}">'
                f'<h2>{result_icons[pred_class]} {pred_label}</h2>'
                f'<p>신뢰도: {pred_conf:.1f}%</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Probability bars
            st.markdown("#### 클래스별 확률")
            prob_cols = st.columns(3)
            for i, (prob_col, name, prob, color) in enumerate(
                zip(prob_cols, CLASS_NAMES, probs, CLASS_COLORS)
            ):
                with prob_col:
                    st.metric(name, f"{prob * 100:.1f}%")
                    st.progress(float(prob))

            # Probability chart (plotly)
            fig_prob = go.Figure(go.Bar(
                y=CLASS_SHORT,
                x=probs * 100,
                orientation="h",
                marker_color=CLASS_COLORS,
                text=[f"{v:.1f}%" for v in probs * 100],
                textposition="outside",
                textfont=dict(size=13),
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig_prob.update_layout(
                title="예측 확률 분포",
                xaxis_title="확률 (%)",
                xaxis=dict(range=[0, 110]),
                height=260,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Risk assessment
            st.divider()
            st.markdown("### 위험도 평가")
            if pred_class == 0:
                st.success(
                    "**낮은 위험** - 이 동물은 생존(입양/반환) 가능성이 높습니다.\n\n"
                    "일반적인 보호 절차를 진행하되, 입양 홍보에 집중하세요."
                )
            elif pred_class == 1:
                st.warning(
                    "**중간 위험** - 자연사 가능성이 감지되었습니다.\n\n"
                    "건강 상태를 면밀히 모니터링하고, 수의사 진료를 권장합니다."
                )
            else:
                st.error(
                    "**높은 위험** - 안락사 위험이 감지되었습니다.\n\n"
                    "**골든타임 확보가 필요합니다.** 긴급 입양 홍보, "
                    "임시보호 연결 등 즉각적인 조치를 권장합니다."
                )

            if not has_model:
                st.caption(
                    "* 데모 모드: 표 데이터 기반 시뮬레이션 결과입니다. "
                    "실제 모델을 로드하면 CNN 이미지 분석이 포함된 정확한 예측이 가능합니다."
                )
