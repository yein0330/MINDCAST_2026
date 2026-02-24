"""데이터 증강 유틸리티 모듈"""
import re
import logging
from typing import List, Set
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def parse_comments(raw_text: str) -> List[str]:
    """LLM 응답에서 개별 댓글 파싱"""
    lines = raw_text.strip().split("\n")
    comments = []
    for line in lines:
        cleaned = re.sub(r"^\s*[\d]+[.)]\s*", "", line)
        cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned)
        cleaned = cleaned.strip().strip('"').strip("'")
        if cleaned:
            comments.append(cleaned)
    return comments


def validate_comment(text: str, config) -> bool:
    """개별 댓글 유효성 검증"""
    if len(text) < config.min_comment_length or len(text) > config.max_comment_length:
        return False

    # 한국어 최소 3자 포함
    korean_chars = len(re.findall(r"[가-힣]", text))
    if korean_chars < 3:
        return False

    # 메타 응답 필터링
    meta_patterns = [
        r"^(네|알겠|감정|댓글|다음|작성)",
        r"(입니다\.|합니다\.)$",
        r"^.{0,5}(번째|번 댓글)",
    ]
    for pattern in meta_patterns:
        if re.search(pattern, text):
            return False

    # 욕설/분노 표현 필터링
    profanity_patterns = [
        r"(시발|씨발|ㅅㅂ|ㅂㅅ|개새끼|미친놈|지랄)",
    ]
    for pattern in profanity_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    return True


def is_duplicate(text: str, existing_texts: List[str], threshold: float = 0.85) -> bool:
    """완전 일치 + 유사도 기반 중복 검사"""
    text_normalized = text.strip().lower()

    for existing in existing_texts:
        existing_normalized = existing.strip().lower()
        if text_normalized == existing_normalized:
            return True
        ratio = SequenceMatcher(None, text_normalized, existing_normalized).ratio()
        if ratio >= threshold:
            return True

    return False


def deduplicate_batch(
    comments: List[dict],
    existing_texts: List[str],
    threshold: float = 0.85,
) -> List[dict]:
    """배치 단위 중복 제거"""
    seen: Set[str] = set(t.lower().strip() for t in existing_texts)
    results = []

    for comment in comments:
        text = comment["text"]
        normalized = text.strip().lower()

        if normalized in seen:
            continue

        # 성능을 위해 최근 500개만 fuzzy 검사
        recent = list(seen)[-500:]
        is_dup = any(
            SequenceMatcher(None, normalized, r).ratio() >= threshold
            for r in recent
        )
        if is_dup:
            continue

        seen.add(normalized)
        results.append(comment)

    removed = len(comments) - len(results)
    if removed > 0:
        logger.info(f"[UTILS] 중복 제거: {removed}개 제거됨")

    return results


def setup_logging(level: str = "INFO") -> None:
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
