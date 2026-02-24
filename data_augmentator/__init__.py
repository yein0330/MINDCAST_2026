"""
Data Augmentator -- 한국어 감정 댓글 데이터 증강 모듈

LLM 에이전트를 활용하여 감정 라벨 불균형을 해소합니다.
전략: 분노(anchor) 기준으로 5감정(슬픔, 불안, 상처, 당황, 기쁨)을
      분노 개수까지 맞추는 anchor-based 생성

사용법:
    python scripts/augment_data.py
    python scripts/augment_data.py --dry-run
"""

from .dataset import AugmentationDataset
from .model import LLMClient, ModelPool
from .agent import AugmentationAgent, GeneratedComment
from .evaluation import AugmentationEvaluator
from .utils import parse_comments, validate_comment, is_duplicate, setup_logging

__all__ = [
    "AugmentationDataset",
    "LLMClient",
    "ModelPool",
    "AugmentationAgent",
    "GeneratedComment",
    "AugmentationEvaluator",
    "parse_comments",
    "validate_comment",
    "is_duplicate",
    "setup_logging",
]
