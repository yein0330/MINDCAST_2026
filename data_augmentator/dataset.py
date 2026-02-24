"""
데이터 증강용 데이터셋 로드 및 분포 분석 모듈

전략:
- 뉴스 타이틀 × 5감정(분노 제외) × 댓글수 균등 생성
- 기존 sentiment_comments 데이터는 감정 분포 참고용
"""

import os
import json
import random
import logging
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

SENTIMENT_LABEL_MAP = {
    "분노": 0, "슬픔": 1, "불안": 2, "상처": 3, "당황": 4, "기쁨": 5,
}
LABEL_ID_TO_NAME = {v: k for k, v in SENTIMENT_LABEL_MAP.items()}

# 🔥 생성 대상 감정 (분노 제외)
GENERATE_EMOTIONS = ["슬픔", "불안", "상처", "당황", "기쁨"]


class AugmentationDataset:
    """증강 대상 데이터셋 관리"""

    def __init__(self, config):
        self.config = config
        self.base_dataset: Optional[Dataset] = None
        self.label_counts: Dict[str, int] = {}
        self._news_titles: Optional[List[str]] = None

    # =====================================================
    # Base sentiment dataset (reference only)
    # =====================================================
    def load_base_data(self) -> Dataset:
        """기존 감정 데이터 로드 (분포 참고용, 분노 제외)"""
        print("[DATA] Loading base sentiment parquet from HuggingFace Hub")

        parquet_path = hf_hub_download(
            repo_id="MindCastSogang/MindCastTrainSet",
            filename="sentiment_comments/train-00000-of-00001.parquet",
            repo_type="dataset",
        )

        df = pd.read_parquet(parquet_path)

        label_col = None
        for c in ["label", "sentiment", "emotion"]:
            if c in df.columns:
                label_col = c
                break

        if label_col is None:
            raise ValueError(f"감정 컬럼을 찾을 수 없습니다: {df.columns}")

        df["label"] = df[label_col].map(SENTIMENT_LABEL_MAP)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        self.base_dataset = Dataset.from_pandas(df, preserve_index=False)

        print(f"[DATA] Loaded {len(self.base_dataset)} samples (reference only)")
        return self.base_dataset

    def get_label_distribution(self) -> Dict[str, int]:
        if self.base_dataset is None:
            raise RuntimeError("load_base_data()를 먼저 호출하세요")

        counter = Counter(self.base_dataset["label"])
        self.label_counts = {
            LABEL_ID_TO_NAME[k]: v
            for k, v in counter.items()
            if k in LABEL_ID_TO_NAME
        }
        return self.label_counts

    # =====================================================
    # News title loader (JSON direct parsing)
    # =====================================================
    def get_news_titles(self) -> List[str]:
        """
        뉴스 제목 리스트 반환
        - hf_hub_download + json.load 사용
        - title 중복 제거
        - timestamp schema 문제 완전 회피
        """
        if self._news_titles is not None:
            return self._news_titles

        if not self.config.news_repo_files:
            logger.warning("[DATA] news_repo_files가 비어있습니다. 샘플 사용")
            self._news_titles = _get_sample_news_titles()
            return self._news_titles

        print(f"[DATA] Loading news titles from {self.config.news_dataset_id}")

        titles = set()

        for rel_path in self.config.news_repo_files:
            local_path = hf_hub_download(
                repo_id=self.config.news_dataset_id,
                filename=rel_path,
                repo_type="dataset",
            )

            with open(local_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            for day in obj.get("data", []):
                for post in day.get("posts", []):
                    title = post.get("title")
                    if title:
                        titles.add(title.strip())

        self._news_titles = sorted(titles)
        print(f"[DATA] Loaded {len(self._news_titles)} unique news titles")
        return self._news_titles

    # =====================================================
    # Generation plan
    # =====================================================
    def get_full_label_distribution(self) -> Dict[str, int]:
        """
        분노 포함 전체 감정 분포 (anchor 기준용)
        """
        parquet_path = hf_hub_download(
            repo_id="MindCastSogang/MindCastTrainSet",
            filename="sentiment_comments/train-00000-of-00001.parquet",
            repo_type="dataset",
        )

        df = pd.read_parquet(parquet_path)

        label_col = None
        for c in ["label", "sentiment", "emotion"]:
            if c in df.columns:
                label_col = c
                break

        if label_col is None:
            raise ValueError("감정 컬럼을 찾을 수 없습니다")

        return dict(Counter(df[label_col]))

    def compute_final_generation_plan_anchor_to_anger(self) -> Dict:
        """
        분노를 기준(anchor)으로
        다른 감정들을 분노 개수까지 맞추는
        최종 실행용 생성 계획 (정수화)
        """

        full_dist = self.get_full_label_distribution()

        if "분노" not in full_dist:
            raise RuntimeError("분노 데이터가 존재하지 않습니다")

        anger_count = full_dist["분노"]

        news_titles = self.get_news_titles()
        num_titles = len(news_titles)

        final_plan = {
            "anchor_emotion": "분노",
            "anchor_count": anger_count,
            "num_titles": num_titles,
            "emotions": {},
            "total_to_generate": 0,
        }

        for emo in GENERATE_EMOTIONS:
            current = full_dist.get(emo, 0)
            need = max(0, anger_count - current)

            # 타이틀당 기본 생성 개수 (floor)
            per_title = need // num_titles
            remainder = need % num_titles

            final_plan["emotions"][emo] = {
                "current": current,
                "target": anger_count,
                "need_to_generate": need,
                "per_title": per_title,
                "extra_titles": remainder,  # 앞에서부터 +1 줄 타이틀 수
            }

            final_plan["total_to_generate"] += need

        return final_plan
    
    def print_final_generation_plan(self, plan: Dict) -> None:
        print("\n" + "=" * 60)
        print("🔥 FINAL GENERATION PLAN (ANGER-ANCHORED)")
        print("=" * 60)

        print(f"Anchor emotion : {plan['anchor_emotion']}")
        print(f"Anchor count  : {plan['anchor_count']}")
        print(f"News titles   : {plan['num_titles']}")
        print("-" * 60)

        for emo, info in plan["emotions"].items():
            print(f"[{emo}]")
            print(f"  current           : {info['current']}")
            print(f"  target            : {info['target']}")
            print(f"  need_to_generate  : {info['need_to_generate']}")
            print(f"  per_title         : {info['per_title']}")
            print(f"  extra_titles (+1) : {info['extra_titles']}")
            print("-" * 60)

        print(f"TOTAL GENERATED COMMENTS: {plan['total_to_generate']}")
        print("=" * 60)

    def compute_generation_plan(self) -> Dict:
        news_titles = self.get_news_titles()
        num_titles = len(news_titles)

        emotions = GENERATE_EMOTIONS
        min_c = self.config.min_comments_per_title
        max_c = self.config.max_comments_per_title

        plan = {
            "num_titles": num_titles,
            "emotions": emotions,
            "per_emotion_min": num_titles * min_c,
            "per_emotion_max": num_titles * max_c,
            "total_min": num_titles * len(emotions) * min_c,
            "total_max": num_titles * len(emotions) * max_c,
        }

        print("\n[PLAN] 생성 계획")
        print(f"  타이틀 수: {num_titles}")
        print(f"  감정: {', '.join(emotions)}")
        print(f"  총 생성량: {plan['total_min']:,} ~ {plan['total_max']:,}")

        return plan

    # =====================================================
    # Few-shot examples
    # =====================================================
    def get_fewshot_examples(self, n: int = 5) -> Dict[str, List[str]]:
        """감정별 few-shot 예시 댓글을 베이스 데이터에서 n개씩 샘플링

        Returns:
            {"슬픔": ["댓글1", "댓글2", ...], "불안": [...], ...}
        """
        parquet_path = hf_hub_download(
            repo_id=self.config.base_dataset_id,
            filename=self.config.base_dataset_parquet,
            repo_type="dataset",
        )
        df = pd.read_parquet(parquet_path)

        examples = {}
        for emotion in self.config.generate_emotions:
            subset = df[df["label"] == emotion]["text"].dropna().tolist()
            # 너무 짧거나 긴 건 제외
            subset = [t.strip() for t in subset if 10 <= len(t.strip()) <= 80]
            if len(subset) > n:
                sampled = random.sample(subset, n)
            else:
                sampled = subset
            examples[emotion] = sampled

        total = sum(len(v) for v in examples.values())
        print(f"[DATA] Few-shot 예시 로드: {total}개 ({n}개 × {len(examples)}감정)")
        return examples

    # =====================================================
    # Utils
    # =====================================================
    def print_reference_distribution(self) -> None:
        if not self.label_counts:
            self.get_label_distribution()

        total = sum(self.label_counts.values())
        print("\n기존 데이터 라벨 분포 (참고)")
        for emo in ["분노","슬픔", "불안", "상처", "당황", "기쁨"]:
            cnt = self.label_counts.get(emo, 0)
            ratio = cnt / total * 100 if total > 0 else 0
            print(f"  {emo}: {cnt} ({ratio:.1f}%)")


def _get_sample_news_titles() -> List[str]:
    return [
        "코로나19 확진자 급증, 정부 긴급 대책 발표",
        "부동산 가격 폭등, 청년층 내 집 마련 더 어려워져",
        "취업난 심화, 청년 실업률 역대 최고치 기록",
        "기후변화 심각, 역대급 폭우로 피해 속출",
    ]
