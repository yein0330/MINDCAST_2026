"""
데이터 증강 설정 모듈
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


# =========================
# Model Endpoint Definition
# =========================

@dataclass(frozen=True)
class ModelEndpoint:
    """개별 모델 API 엔드포인트 설정"""
    name: str
    provider: str               # openai | google | huggingface | groq
    model_id: str
    api_key_env: str            # 환경변수명
    base_url: Optional[str] = None

    max_tokens: int = 256
    temperature: float = 0.9
    requests_per_minute: int = 60


# =========================
# Augmentation Config
# =========================

@dataclass
class AugmentationConfig:
    """
    데이터 증강 전체 설정
    """

    # ---------------------
    # Emotion definitions
    # ---------------------
    ALL_EMOTIONS: List[str] = field(default_factory=lambda: [
        "분노", "슬픔", "불안", "상처", "당황", "기쁨"
    ])

    generate_emotions: List[str] = field(default_factory=lambda: [
        "슬픔", "불안", "상처", "당황", "기쁨"
    ])

    anchor_emotion: str = "분노"

    # ---------------------
    # Dataset sources
    # ---------------------
    base_dataset_id: str = "MindCastSogang/MindCastTrainSet"
    base_dataset_parquet: str = "sentiment_comments/train-00000-of-00001.parquet"

    news_dataset_id: str = "MindCastSogang/Youtube_news_preprocessed_data"
    news_data_dir: str = "preprocessed/v1"
    news_repo_files: List[str] = field(default_factory=lambda: [
        "preprocessed/v1/2020/01/01-10/news_comments.json",
        "preprocessed/v1/2020/01/11-20/news_comments.json",
        "preprocessed/v1/2020/01/21-31/news_comments.json",
    ])

    output_dataset_id: str = "MindCastSogang/mindcast-augmented-sc"

    hf_token_env: str = "HF_TOKEN"

    # ---------------------
    # Generation strategy
    # ---------------------
    use_anchor_based_plan: bool = True

    min_comments_per_title: int = 5
    max_comments_per_title: int = 10

    comments_per_prompt: int = 5
    allow_partial_prompt: bool = True

    # ---------------------
    # Quality constraints
    # ---------------------
    min_comment_length: int = 15
    max_comment_length: int = 60
    duplicate_threshold: float = 0.85

    # ---------------------
    # Retry & checkpoint
    # ---------------------
    max_retries: int = 3
    retry_delay: float = 2.0

    save_interval: int = 100
    checkpoint_dir: str = "augmentation_checkpoints"

    # ---------------------
    # Model endpoints
    # ---------------------
    model_endpoints: List[ModelEndpoint] = field(default_factory=lambda: [
        ModelEndpoint(
            name="gpt4o-mini",
            provider="openai",
            model_id="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            requests_per_minute=500,
        ),
        ModelEndpoint(
            name="gemini-2.0-flash-lite",
            provider="google",
            model_id="gemini-2.0-flash-lite",
            api_key_env="GEMINI_API_KEY",
            requests_per_minute=1000,
        ),
        ModelEndpoint(
            name="llama3-8b",
            provider="huggingface",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            api_key_env="HF_TOKEN",
            base_url="https://router.huggingface.co/v1",
            requests_per_minute=60,
        ),
        ModelEndpoint(
            name="gemma3-27b",
            provider="huggingface",
            model_id="google/gemma-3-27b-it",
            api_key_env="HF_TOKEN",
            base_url="https://router.huggingface.co/v1",
            requests_per_minute=60,
        ),
        ModelEndpoint(
            name="qwen2.5-7b",
            provider="huggingface",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            api_key_env="HF_TOKEN",
            base_url="https://router.huggingface.co/v1",
            requests_per_minute=60,
        ),
    ])

    # ---------------------
    # Emotion → Model routing
    # ---------------------
    emotion_model_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "슬픔": {
            "gpt4o-mini": 0.30,
            "gemini-2.0-flash-lite": 0.25,
            "llama3-8b": 0.20,
            "gemma3-27b": 0.15,
            "qwen2.5-7b": 0.10,
        },
        "불안": {
            "gemini-2.0-flash-lite": 0.30,
            "gpt4o-mini": 0.25,
            "llama3-8b": 0.20,
            "qwen2.5-7b": 0.15,
            "gemma3-27b": 0.10,
        },
        "상처": {
            "llama3-8b": 0.30,
            "gpt4o-mini": 0.25,
            "gemini-2.0-flash-lite": 0.20,
            "gemma3-27b": 0.15,
            "qwen2.5-7b": 0.10,
        },
        "당황": {
            "qwen2.5-7b": 0.30,
            "gemini-2.0-flash-lite": 0.25,
            "llama3-8b": 0.20,
            "gpt4o-mini": 0.15,
            "gemma3-27b": 0.10,
        },
        "기쁨": {
            "gemma3-27b": 0.30,
            "qwen2.5-7b": 0.25,
            "llama3-8b": 0.20,
            "gpt4o-mini": 0.15,
            "gemini-2.0-flash-lite": 0.10,
        },
    })

    # ---------------------
    # Personas
    # ---------------------
    personas: List[Dict] = field(default_factory=lambda: [
        {
            "type": "직장인",
            "description": "30대 회사원, 야근이 많고 스트레스를 받는 직장인",
            "traits": ["현실적", "피곤한", "공감 능력 높은"],
        },
        {
            "type": "대학생",
            "description": "20대 대학생, 취업 준비와 학업에 쫓기는 학생",
            "traits": ["감정적", "솔직한", "유행에 민감한"],
        },
        {
            "type": "주부",
            "description": "40대 전업주부, 가정과 육아에 헌신적인 어머니",
            "traits": ["걱정 많은", "보호적인", "감수성 풍부한"],
        },
        {
            "type": "은퇴자",
            "description": "60대 은퇴자, 사회 변화에 관심이 많은 시니어",
            "traits": ["경험 많은", "보수적", "걱정이 많은"],
        },
    ])

    # ---------------------
    # Metadata & dry-run
    # ---------------------
    save_metadata: bool = True
    metadata_fields: List[str] = field(default_factory=lambda: [
        "model",
        "persona_type",
        "emotion",
        "news_title",
        "prompt_id",
    ])

    dry_run: bool = False
    dry_run_titles: int = 5

    log_level: str = "INFO"

    # ---------------------
    # Validation helpers
    # ---------------------
    def validate(self):
        assert self.anchor_emotion not in self.generate_emotions
        for e in self.generate_emotions:
            assert e in self.emotion_model_weights, f"Missing weights for {e}"

    def check_env(self):
        missing = []
        for m in self.model_endpoints:
            if not os.getenv(m.api_key_env):
                missing.append(m.api_key_env)
        if missing:
            raise EnvironmentError(f"Missing API keys: {missing}")
