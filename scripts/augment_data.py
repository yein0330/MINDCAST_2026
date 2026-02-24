#!/usr/bin/env python3
"""
데이터 증강 실행 스크립트

전략: 분노(anchor) 기준으로 5감정(슬픔, 불안, 상처, 당황, 기쁨)을
      분노 개수까지 맞추는 anchor-based 생성

사용법:
    python scripts/augment_data.py                    # 전체 증강 실행
    python scripts/augment_data.py --dry-run           # 생성 계획만 출력
    python scripts/augment_data.py --push              # 생성 후 HF 업로드
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.augmentation_config import AugmentationConfig
from data_augmentator.dataset import AugmentationDataset
from data_augmentator.model import ModelPool
from data_augmentator.agent import AugmentationAgent
from data_augmentator.evaluation import AugmentationEvaluator
from data_augmentator.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="MindCast 데이터 증강 (anchor-based 균등 생성)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 생성 없이 계획만 출력",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="생성 후 HuggingFace Hub에 업로드",
    )
    args = parser.parse_args()

    # 1. Config
    config = AugmentationConfig()
    config.validate()
    setup_logging(config.log_level)

    print("=" * 60)
    print("MindCast 데이터 증강 파이프라인 (Anchor-Based)")
    print(f"Anchor: {config.anchor_emotion}")
    print(f"생성 대상: {', '.join(config.generate_emotions)}")
    print("=" * 60)

    # 2. Dataset 로드 및 분포 분석
    dataset = AugmentationDataset(config)
    dataset.load_base_data()
    dataset.get_label_distribution()
    dataset.print_reference_distribution()

    # 3. Anchor-based 생성 계획 산출
    plan = dataset.compute_final_generation_plan_anchor_to_anger()
    dataset.print_final_generation_plan(plan)

    if args.dry_run:
        print("\n[DRY-RUN] --dry-run 모드. 실제 생성 없이 종료합니다.")
        return

    # 4. Model pool 초기화
    model_pool = ModelPool(config)
    if not model_pool.clients:
        print("[ERROR] 사용 가능한 모델이 없습니다. API 키를 확인하세요.")
        print("  필요 환경변수: OPENAI_API_KEY, GEMINI_API_KEY, "
              "HF_TOKEN, GROQ_API_KEY")
        sys.exit(1)
    print(f"\n[MODEL] 활성 모델 {len(model_pool.clients)}개: "
          f"{model_pool.get_active_model_names()}")

    # 5. Agent로 댓글 생성 (anchor-based plan 전달)
    agent = AugmentationAgent(config, model_pool, dataset)
    generated = agent.run(plan)
    stats = agent.get_stats()

    print(f"\n[RESULT] 총 {len(generated):,}개 댓글 생성 완료")
    print("\n감정별:")
    for emotion, count in stats.get("by_emotion", {}).items():
        print(f"  {emotion}: {count:,}개")
    print("\n모델별:")
    for model, count in stats.get("by_model", {}).items():
        print(f"  {model}: {count:,}개")

    # 6. 평가
    original_counts = dataset.get_label_distribution()
    generated_dicts = [
        {
            "text": c.text,
            "label": c.label,
            "emotion": c.emotion,
            "source_model": c.source_model,
            "persona_type": c.persona_type,
        }
        for c in generated
    ]

    evaluator = AugmentationEvaluator(config)
    evaluator.print_generated_report(generated_dicts, original_counts, plan)

    # 7. HF Hub 업로드 (선택)
    if args.push:
        dataset.push_to_hub(generated_dicts)
        print(f"\n[HF] 데이터셋 업로드 완료: {config.output_dataset_id}")

    print("\n" + "=" * 60)
    print("데이터 증강 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
