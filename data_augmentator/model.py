"""LLM API 통합 인터페이스 모듈"""
import os
import time
import random
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """단일 API 호출 결과"""
    raw_text: str = ""
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error: Optional[str] = None


class LLMClient:
    """단일 모델 API 클라이언트"""

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.api_key = os.environ.get(endpoint.api_key_env, "")
        self._last_call_time = 0.0
        self._min_interval = 60.0 / endpoint.requests_per_minute
        self._client = None

    def _get_client(self):
        """Provider별 클라이언트 초기화 (lazy)"""
        if self._client is not None:
            return self._client

        provider = self.endpoint.provider

        if provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.endpoint.model_id)
        else:
            # openai, huggingface, groq 모두 OpenAI 호환 API
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.endpoint.base_url:
                kwargs["base_url"] = self.endpoint.base_url
            self._client = OpenAI(**kwargs)

        return self._client

    def _wait_for_rate_limit(self):
        """Rate limit 준수를 위한 대기"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

    def generate(self, prompt: str) -> GenerationResult:
        """단일 프롬프트로 텍스트 생성 (rate limiting + retry 포함)"""
        client = self._get_client()
        last_error = None

        for attempt in range(3):
            self._wait_for_rate_limit()
            try:
                if self.endpoint.provider == "google":
                    response = client.generate_content(
                        prompt,
                        generation_config={
                            "max_output_tokens": self.endpoint.max_tokens,
                            "temperature": self.endpoint.temperature,
                        },
                    )
                    raw_text = response.text
                    return GenerationResult(
                        raw_text=raw_text,
                        model_name=self.endpoint.name,
                        success=True,
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.endpoint.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.endpoint.max_tokens,
                        temperature=self.endpoint.temperature,
                    )
                    raw_text = response.choices[0].message.content or ""
                    usage = response.usage
                    return GenerationResult(
                        raw_text=raw_text,
                        model_name=self.endpoint.name,
                        prompt_tokens=usage.prompt_tokens if usage else 0,
                        completion_tokens=usage.completion_tokens if usage else 0,
                        success=True,
                    )

            except Exception as e:
                last_error = str(e)
                error_str = last_error.lower()

                # 인증/결제 오류는 즉시 실패 (재시도 무의미)
                if any(code in error_str for code in [
                    "401", "unauthorized", "402", "credit balance",
                    "insufficient_quota",
                ]):
                    logger.error(f"[{self.endpoint.name}] 복구 불가 오류: {last_error}")
                    break

                # Rate limit 또는 서버 에러는 재시도
                delay = (2 ** attempt) * 2
                logger.warning(
                    f"[{self.endpoint.name}] 오류 (시도 {attempt + 1}/3): "
                    f"{last_error}. {delay}초 후 재시도..."
                )
                time.sleep(delay)

        return GenerationResult(
            model_name=self.endpoint.name,
            success=False,
            error=last_error,
        )

    def is_available(self) -> bool:
        """API 키가 설정되어 있는지 확인"""
        return bool(self.api_key)


class ModelPool:
    """모든 모델 클라이언트를 관리하는 풀"""

    def __init__(self, config):
        self.clients: List[LLMClient] = []
        for endpoint in config.model_endpoints:
            client = LLMClient(endpoint)
            if client.is_available():
                self.clients.append(client)
                print(f"[MODEL] {endpoint.name} 활성화")
            else:
                print(f"[MODEL] {endpoint.name} 비활성화 "
                      f"({endpoint.api_key_env} 환경변수 미설정)")

    def generate_round_robin(self, prompts: List[str]) -> List[GenerationResult]:
        """프롬프트들을 모델들에 라운드로빈으로 분배하여 생성"""
        if not self.clients:
            return [
                GenerationResult(success=False, error="사용 가능한 모델 없음")
                for _ in prompts
            ]

        results = []
        for i, prompt in enumerate(prompts):
            client = self.clients[i % len(self.clients)]
            result = client.generate(prompt)
            results.append(result)

        return results

    def generate_with_model(self, model_name: str, prompt: str) -> GenerationResult:
        """특정 모델로 생성"""
        for client in self.clients:
            if client.endpoint.name == model_name:
                return client.generate(prompt)

        return GenerationResult(
            success=False,
            error=f"모델 '{model_name}'을 찾을 수 없음",
        )

    def generate_weighted(
        self, prompt: str, weights: Dict[str, float]
    ) -> GenerationResult:
        """감정별 가중치에 따라 모델을 선택하여 생성

        Args:
            prompt: 생성 프롬프트
            weights: {model_name: weight} 가중치 딕셔너리
        """
        if not self.clients:
            return GenerationResult(success=False, error="사용 가능한 모델 없음")

        available = {c.endpoint.name: c for c in self.clients}
        candidates = []
        candidate_weights = []

        for name, weight in weights.items():
            if name in available:
                candidates.append(available[name])
                candidate_weights.append(weight)

        if not candidates:
            # 가중치에 해당하는 모델이 없으면 활성 모델 중 랜덤 선택
            selected = random.choice(self.clients)
        else:
            selected = random.choices(
                candidates, weights=candidate_weights, k=1
            )[0]

        return selected.generate(prompt)

    def get_active_model_names(self) -> List[str]:
        """활성화된 모델 이름 목록"""
        return [c.endpoint.name for c in self.clients]
