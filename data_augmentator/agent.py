"""데이터 증강 에이전트 모듈 -- 페르소나 기반 댓글 생성

전략: 분노(anchor) 기준 anchor-based 생성.
      감정별로 타이틀을 모델에 가중치 비율로 블록 할당.
      각 모델은 담당 타이틀에 대해 페르소나를 순환하며 생성.
"""
import os
import math
import logging
from typing import Dict, List
from dataclasses import dataclass

import pandas as pd

from .dataset import SENTIMENT_LABEL_MAP
from .utils import parse_comments, validate_comment, is_duplicate

logger = logging.getLogger(__name__)


@dataclass
class GeneratedComment:
    """생성된 단일 댓글"""
    text: str
    emotion: str
    persona_type: str
    news_title: str
    source_model: str
    label: int


class AugmentationAgent:
    """페르소나 기반 댓글 생성 에이전트

    감정별로 타이틀을 모델에 블록 할당하고,
    각 모델이 담당 타이틀에 대해 페르소나를 순환하며 댓글을 생성한다.
    """

    def __init__(self, config, model_pool, dataset):
        self.config = config
        self.model_pool = model_pool
        self.dataset = dataset
        self.generated: List[GeneratedComment] = []
        # 감정별 few-shot 예시 (베이스 데이터에서 로드)
        self.fewshot_examples: Dict[str, List[str]] = dataset.get_fewshot_examples(n=5)

    # =========================================================
    # Title → Model 블록 할당
    # =========================================================
    def _assign_titles_to_models(
        self, num_titles: int, emotion_weights: Dict[str, float]
    ) -> Dict[str, List[int]]:
        """타이틀을 모델에 가중치 비율로 블록 할당

        Returns:
            {model_name: [title_index, ...]}
        """
        active_names = self.model_pool.get_active_model_names()

        # 활성 모델 중 가중치가 있는 것만 필터
        filtered = {
            name: emotion_weights[name]
            for name in active_names
            if name in emotion_weights and emotion_weights[name] > 0
        }

        if not filtered:
            # 가중치 매칭 없으면 활성 모델 균등 분배
            filtered = {name: 1.0 for name in active_names}

        total_w = sum(filtered.values())
        models = list(filtered.keys())

        assignments = {}
        cursor = 0

        for i, model_name in enumerate(models):
            if i == len(models) - 1:
                # 마지막 모델이 나머지 흡수
                block_size = num_titles - cursor
            else:
                block_size = round(num_titles * filtered[model_name] / total_w)

            assignments[model_name] = list(range(cursor, cursor + block_size))
            cursor += block_size

        return assignments

    # =========================================================
    # Prompt
    # =========================================================
    def build_prompt(self, persona: Dict, emotion: str, news_title: str) -> str:
        n_comments = self.config.comments_per_prompt
        persona_type = persona["type"]
        traits = ', '.join(persona['traits'])

        # few-shot 예시 블록 구성
        examples = self.fewshot_examples.get(emotion, [])
        if examples:
            example_lines = '\n'.join(f'- {ex}' for ex in examples)
            example_block = f"""
아래는 '{emotion}' 감정이 담긴 실제 댓글 예시입니다. 이런 톤과 감정 수준을 참고하세요:
{example_lines}
"""
        else:
            example_block = ""

        return f"""당신은 '{persona_type}'입니다.
구체적 설정: {persona['description']}
말투 특성: {traits}

{persona_type}으로서 평소 습관대로 뉴스 댓글을 작성합니다.
{persona_type}만의 경험, 상황, 어투가 댓글에 자연스럽게 녹아들어야 합니다.
{example_blok}
뉴스 제목: {news_title}

위 뉴스에 대해 '{emotion}' 감정이 드러나는 댓글을 {n_comments}개 작성하세요.

규칙:
1. {persona_type}의 일상·상황이 구체적으로 드러나야 합니다 (예: 직장인이면 야근·회의·상사 언급, 대학생이면 과제·취준·등록금 언급, 주부면 아이·장보기·살림 언급, 은퇴자면 연금·건강·손주 언급)
2. '{emotion}' 감정이 명확히 표현되어야 합니다
3. 각 댓글은 15~60자
4. 분노, 욕설, 과도한 비난은 금지
5. 실제 한국인이 쓴 것처럼 자연스럽게 (줄임말, 이모티콘 자유)
6. 댓글끼리 서로 다른 표현과 관점을 사용하세요

형식: 각 댓글을 줄바꿈으로 구분 (번호 없이)"""

    # =========================================================
    # 단일 타이틀 생성 (지정 모델 사용)
    # =========================================================
    def _generate_for_title(
        self,
        news_title: str,
        emotion: str,
        target_count: int,
        model_name: str,
        persona_offset: int,
    ) -> List[GeneratedComment]:
        """특정 타이틀에 대해 지정된 모델로 target_count만큼 댓글 생성

        페르소나를 균등 분배: target_count를 4개 페르소나에 나눠서
        각 페르소나별로 별도 호출하여 다양성 보장.

        Args:
            persona_offset: 페르소나 순환 시작 오프셋 (타이틀 로컬 인덱스)
        """
        results = []
        existing_texts = [c.text for c in self.generated]
        personas = self.config.personas
        num_personas = len(personas)

        # 페르소나별 할당량 계산
        base_per_persona = target_count // num_personas
        remainder = target_count % num_personas

        for p_idx in range(num_personas):
            persona = personas[(persona_offset + p_idx) % num_personas]
            persona_target = base_per_persona + (1 if p_idx < remainder else 0)
            if persona_target <= 0:
                continue

            max_retries = 3
            for attempt in range(max_retries):
                prompt = self.build_prompt(persona, emotion, news_title)
                api_result = self.model_pool.generate_with_model(
                    model_name, prompt
                )

                if not api_result.success:
                    logger.warning(
                        f"[AGENT] API 실패 ({model_name}): {api_result.error}"
                    )
                    continue

                persona_collected = sum(
                    1 for r in results if r.persona_type == persona["type"]
                )

                comments = parse_comments(api_result.raw_text)
                for text in comments:
                    if persona_collected >= persona_target:
                        break
                    if not validate_comment(text, self.config):
                        continue

                    all_existing = existing_texts + [r.text for r in results]
                    if is_duplicate(
                        text, all_existing, self.config.duplicate_threshold
                    ):
                        continue

                    results.append(GeneratedComment(
                        text=text,
                        emotion=emotion,
                        persona_type=persona["type"],
                        news_title=news_title,
                        source_model=model_name,
                        label=SENTIMENT_LABEL_MAP[emotion],
                    ))
                    persona_collected += 1

                if persona_collected >= persona_target:
                    break

        return results[:target_count]

    # =========================================================
    # 메인 실행
    # =========================================================
    def run(self, plan: Dict) -> List[GeneratedComment]:
        """
        Anchor-based 증강 파이프라인 실행.

        흐름:
        1. 감정별로 타이틀을 모델에 블록 할당
        2. 각 모델이 담당 타이틀 순회
        3. 타이틀당 ceil(per_title / comments_per_prompt)회 호출
        4. 호출마다 페르소나 순환
        """
        news_titles = self.dataset.get_news_titles()
        cpp = self.config.comments_per_prompt

        print(f"\n[AGENT] Anchor-based 생성 시작")
        print(f"  타이틀: {plan['num_titles']}개")
        print(f"  감정: {len(plan['emotions'])}개")
        print(f"  comments_per_prompt: {cpp}")
        print(f"  총 생성 목표: {plan['total_to_generate']:,}개")

        for emotion, info in plan["emotions"].items():
            need = info["need_to_generate"]
            if need <= 0:
                print(
                    f"[AGENT] [{emotion}] 이미 충분 "
                    f"(현재 {info['current']} >= 목표 {info['target']})"
                )
                continue

            per_title = info["per_title"]
            extra_titles = info["extra_titles"]
            emotion_weights = self.config.emotion_model_weights.get(emotion, {})

            # 타이틀 블록 할당
            assignments = self._assign_titles_to_models(
                len(news_titles), emotion_weights
            )

            # 할당 결과 출력
            print(f"\n[AGENT] [{emotion}] 생성 시작: {need}개 목표 "
                  f"(타이틀당 {per_title}개, 앞 {extra_titles}개 +1)")
            for m_name, t_indices in assignments.items():
                calls_est = sum(
                    math.ceil(
                        (per_title + (1 if ti < extra_titles else 0)) / cpp
                    )
                    for ti in t_indices
                    if per_title + (1 if ti < extra_titles else 0) > 0
                )
                print(f"  {m_name}: {len(t_indices)}개 타이틀 → ~{calls_est}회 호출")

            # 모델별 담당 타이틀 순회
            for model_name, title_indices in assignments.items():
                for local_idx, title_idx in enumerate(title_indices):
                    target_count = per_title + (
                        1 if title_idx < extra_titles else 0
                    )
                    if target_count <= 0:
                        continue

                    results = self._generate_for_title(
                        news_titles[title_idx],
                        emotion,
                        target_count,
                        model_name,
                        local_idx,
                    )
                    self.generated.extend(results)

                # 모델 단위 진행 출력
                emotion_count = sum(
                    1 for c in self.generated if c.emotion == emotion
                )
                print(
                    f"  [{emotion}] {model_name} 완료 "
                    f"({len(title_indices)}개 타이틀, "
                    f"누적 {emotion_count}개)"
                )

            emotion_total = sum(
                1 for c in self.generated if c.emotion == emotion
            )
            print(f"[AGENT] [{emotion}] 완료: {emotion_total}개 생성")

            # 감정별 체크포인트
            self._save_checkpoint()

        print(f"\n[AGENT] 전체 생성 완료: {len(self.generated):,}개")
        return self.generated

    # =========================================================
    # 체크포인트 & 통계
    # =========================================================
    def _save_checkpoint(self):
        """체크포인트 저장"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{len(self.generated)}.csv",
        )

        data = [
            {
                "text": c.text,
                "label": c.label,
                "emotion": c.emotion,
                "source_model": c.source_model,
                "persona_type": c.persona_type,
                "news_title": c.news_title,
            }
            for c in self.generated
        ]
        pd.DataFrame(data).to_csv(
            checkpoint_path, index=False, encoding="utf-8-sig"
        )
        print(f"[CHECKPOINT] {len(self.generated):,}개 저장 → {checkpoint_path}")

    def get_stats(self) -> Dict:
        """생성 통계 반환"""
        by_emotion = {}
        by_model = {}
        by_persona = {}

        for c in self.generated:
            by_emotion[c.emotion] = by_emotion.get(c.emotion, 0) + 1
            by_model[c.source_model] = by_model.get(c.source_model, 0) + 1
            by_persona[c.persona_type] = by_persona.get(c.persona_type, 0) + 1

        return {
            "total": len(self.generated),
            "by_emotion": by_emotion,
            "by_model": by_model,
            "by_persona": by_persona,
        }
