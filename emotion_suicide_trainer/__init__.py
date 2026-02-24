"""
Emotion-Mediated Suicide Forecasting Trainer

2단계 자살 예측 모델 (TwoStageSuicideModel)

이론적 프레임워크:
  가정 1: Macro stress → Emotional amplification  (Stage 1: EmotionAmplifier)
  가정 2: Emotional amplification → Suicide variation (Stage 2: SuicideForecaster)

모듈 구성:
  utils              : 시드, 폰트, CSV 로드, Config 상수
  dataset            : FullLCombinationDataset (시계열 데이터셋)
  model              : EmotionAmplifier / SuicideForecaster / TwoStageSuicideModel
  data_loader        : HuggingFace 데이터 로드 + base_data.csv 생성 (CLI 전용)
  trainer            : 학습/평가 루프 및 메인 파이프라인
  supervised_pipeline: Stage1→Stage2→Joint 3단계 학습
  evaluation         : Attention 시각화, Feature Importance, Sensitivity 분석

사용법:
    # 데이터 로드 (torch/matplotlib 불필요)
    python -m emotion_suicide_trainer.data_loader --inspect --token hf_xxx

    # 학습
    python -m emotion_suicide_trainer.trainer
    python -m emotion_suicide_trainer.supervised_pipeline
"""

# ── 가벼운 모듈만 패키지 임포트 시 즉시 로드 ──────────────────────────────
# torch / matplotlib / seaborn이 없어도 data_loader 단독 실행 가능하도록
# trainer, evaluation 등 heavy 모듈은 여기서 임포트하지 않음.

from .utils import load_clean_csv                        # torch-free
from .dataset import FullLCombinationDataset, EMOTION_COLS_DEFAULT  # numpy only
from .model import (                                     # torch 필요
    EMOTION_NAMES,
    SinusoidalPositionalEncoding,
    EmotionAmplifier,
    SuicideForecaster,
    EmotionSequenceForecaster,
    TwoStageSuicideModel,
)

# data_loader는 패키지 공개 API에서 제외 — CLI 전용 스크립트
# from .data_loader import ...   ← 임포트 안 함

__version__ = "1.0.0"
__all__ = [
    # Utils
    "load_clean_csv",
    # Dataset
    "FullLCombinationDataset", "EMOTION_COLS_DEFAULT",
    # Model
    "EMOTION_NAMES",
    "SinusoidalPositionalEncoding",
    "EmotionAmplifier",
    "SuicideForecaster",
    "EmotionSequenceForecaster",
    "TwoStageSuicideModel",
]
