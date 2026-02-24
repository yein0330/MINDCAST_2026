"""생성 데이터 품질 및 분포 평가 모듈"""
import re
import logging
import numpy as np
from typing import Dict, List
from collections import Counter

logger = logging.getLogger(__name__)


class AugmentationEvaluator:
    """증강 결과 평가"""

    def __init__(self, config):
        self.config = config

    def evaluate_length_distribution(self, comments: List[str]) -> Dict:
        """댓글 길이 분포 분석"""
        lengths = [len(c) for c in comments]
        if not lengths:
            return {"count": 0}

        arr = np.array(lengths)
        return {
            "count": len(lengths),
            "mean": round(float(arr.mean()), 1),
            "median": round(float(np.median(arr)), 1),
            "std": round(float(arr.std()), 1),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "q25": round(float(np.percentile(arr, 25)), 1),
            "q75": round(float(np.percentile(arr, 75)), 1),
        }

    def evaluate_diversity(self, comments: List[str]) -> Dict:
        """어휘 다양성 평가 (Type-Token Ratio)"""
        all_words = []
        for comment in comments:
            words = re.findall(r"[가-힣]+", comment)
            all_words.extend(words)

        if not all_words:
            return {"ttr": 0.0, "unique_words": 0, "total_words": 0}

        unique_words = set(all_words)
        ttr = len(unique_words) / len(all_words)

        return {
            "ttr": round(ttr, 4),
            "unique_words": len(unique_words),
            "total_words": len(all_words),
        }

    def evaluate_by_model(self, comments: List[Dict]) -> Dict:
        """모델별 생성 품질 비교"""
        by_model = {}
        for c in comments:
            model = c.get("source_model", "unknown")
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(c["text"])

        result = {}
        for model, texts in by_model.items():
            lengths = [len(t) for t in texts]
            result[model] = {
                "count": len(texts),
                "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
                "diversity": self.evaluate_diversity(texts),
            }

        return result

    def evaluate_by_persona(self, comments: List[Dict]) -> Dict:
        """페르소나별 생성 통계"""
        by_persona = Counter(c.get("persona_type", "unknown") for c in comments)
        return dict(by_persona)

    def print_generated_report(
        self,
        generated_comments: List[Dict],
        original_counts: Dict[str, int],
        plan: Dict,
    ) -> None:
        """anchor-based plan 기반 종합 평가 리포트 출력"""
        print("\n" + "=" * 60)
        print("데이터 증강 평가 리포트 (Anchor-Based)")
        print("=" * 60)

        # 감정별 생성 결과 vs 목표
        gen_by_emotion = Counter(c["emotion"] for c in generated_comments)

        print(f"\n[1] 감정별 생성 결과 (anchor: {plan['anchor_emotion']}, "
              f"목표: {plan['anchor_count']})")
        print(f"{'감정':6s} | {'기존':>8s} | {'생성':>8s} | {'합계':>8s} | {'목표':>8s}")
        print("-" * 55)

        for emotion, info in plan["emotions"].items():
            generated_count = gen_by_emotion.get(emotion, 0)
            total = info["current"] + generated_count
            print(f"{emotion:6s} | {info['current']:>8d} | "
                  f"{generated_count:>8d} | {total:>8d} | {info['target']:>8d}")

        # 균형 점수
        final_counts = {}
        final_counts[plan["anchor_emotion"]] = plan["anchor_count"]
        for emotion, info in plan["emotions"].items():
            final_counts[emotion] = info["current"] + gen_by_emotion.get(emotion, 0)

        balance = self.compute_balance_score(final_counts)
        orig_balance = self.compute_balance_score(original_counts)
        print(f"\n균형 점수: {orig_balance:.4f} → {balance:.4f}")

        if not generated_comments:
            print("\n" + "=" * 60)
            return

        texts = [c["text"] for c in generated_comments]

        # 길이 분포
        length_stats = self.evaluate_length_distribution(texts)
        print(f"\n[2] 생성 댓글 길이 분포")
        print(f"  평균: {length_stats['mean']}자, 중앙값: {length_stats['median']}자, "
              f"표준편차: {length_stats['std']}자")
        print(f"  범위: {length_stats['min']}~{length_stats['max']}자 "
              f"(Q25={length_stats['q25']}, Q75={length_stats['q75']})")

        # 다양성
        diversity = self.evaluate_diversity(texts)
        print(f"\n[3] 어휘 다양성")
        print(f"  TTR: {diversity['ttr']:.4f} "
              f"({diversity['unique_words']}개 고유 단어 / "
              f"{diversity['total_words']}개 총 단어)")

        # 모델별
        model_stats = self.evaluate_by_model(generated_comments)
        print(f"\n[4] 모델별 통계")
        for model, stats in model_stats.items():
            print(f"  {model}: {stats['count']}개, "
                  f"평균 {stats['avg_length']}자, "
                  f"TTR={stats['diversity']['ttr']:.4f}")

        # 페르소나별
        persona_stats = self.evaluate_by_persona(generated_comments)
        print(f"\n[5] 페르소나별 통계")
        for persona, count in sorted(persona_stats.items(), key=lambda x: -x[1]):
            print(f"  {persona}: {count}개")

        print("\n" + "=" * 60)

    @staticmethod
    def compute_balance_score(counts: Dict[str, int]) -> float:
        """라벨 균형 점수 계산 (0~1, 1=완전 균형)"""
        if not counts:
            return 0.0
        values = list(counts.values())
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        std = np.std(values)
        score = max(0.0, 1.0 - (std / mean))
        return round(score, 4)
