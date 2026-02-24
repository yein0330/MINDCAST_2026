# Data Augmentator - 한국어 감정 댓글 데이터 증강

LLM 에이전트 기반 감정 분류 데이터 증강 파이프라인.

기존 데이터셋의 심각한 라벨 불균형(분노 65% vs 상처 1%)을 해소하기 위해,
분노(anchor) 개수를 기준으로 나머지 5개 감정의 댓글을 LLM으로 생성합니다.

## 전략: Anchor-Based Generation

```
기존 분포:
  분노: 1,036 (65.7%)  ← anchor
  슬픔:   103 ( 6.5%)
  불안:    42 ( 2.7%)
  상처:    15 ( 1.0%)
  당황:   223 (14.1%)
  기쁨:   159 (10.1%)

→ 각 감정을 분노 수준(1,036개)까지 생성
→ 총 4,638개 댓글 생성
→ 6개 감정 모두 ~1,036개로 균등화
```

## 파일 구조

```
data_augmentator/
├── .env                  # API 키 (git에 포함하지 않음)
├── __init__.py
├── agent.py              # 생성 에이전트 (타이틀 블록 할당 + 페르소나 순환)
├── dataset.py            # 데이터셋 로드, 분포 분석, anchor-based plan 산출
├── evaluation.py         # 생성 결과 평가 (분포, 다양성, 길이, 모델별/페르소나별)
├── model.py              # LLM API 통합 인터페이스 (OpenAI/Google/HuggingFace/Groq)
├── utils.py              # 파싱, 검증, 중복 제거 유틸리티
└── README.md

configs/
└── augmentation_config.py  # 전체 설정 (모델, 감정, 페르소나, 가중치)

scripts/
└── augment_data.py         # CLI 진입점

run/
└── run_data_augmentation.sh  # 실행 스크립트
```

## 빠른 시작

### 1. API 키 설정

`data_augmentator/.env` 파일에 API 키를 설정합니다:

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
HF_TOKEN=hf_...
# GROQ_API_KEY=gsk_...  (선택)
```

설정된 키에 해당하는 모델만 활성화되고, 미설정 모델은 자동 스킵됩니다.

### 2. 생성 계획 확인 (Dry Run)

API 호출 없이 현재 데이터 분포와 생성 계획만 확인합니다:

```bash
# 방법 1: 스크립트 직접 실행
cd mindcastlib_trainer
source data_augmentator/.env
python scripts/augment_data.py --dry-run

# 방법 2: 실행 스크립트 사용
bash run/run_data_augmentation.sh
```

출력 예시:

```
============================================================
🔥 FINAL GENERATION PLAN (ANGER-ANCHORED)
============================================================
Anchor emotion : 분노
Anchor count  : 1036
News titles   : 150
------------------------------------------------------------
[슬픔]
  current           : 103
  target            : 1036
  need_to_generate  : 933
  per_title         : 6
  extra_titles (+1) : 33
------------------------------------------------------------
...
TOTAL GENERATED COMMENTS: 4638
============================================================
```

### 3. 실제 생성 실행

```bash
source data_augmentator/.env
python scripts/augment_data.py
```

### 4. HuggingFace Hub 업로드 (선택)

```bash
python scripts/augment_data.py --push
```

## 모델 구성

| 모델 | Provider | API 키 | RPM |
|------|----------|--------|-----|
| gpt-4o-mini | OpenAI | `OPENAI_API_KEY` | 500 |
| gemini-2.0-flash-lite | Google | `GEMINI_API_KEY` | 1000 |
| Mistral-7B-Instruct | HuggingFace Inference | `HF_TOKEN` | 60 |
| gemma-2-2b-it | HuggingFace Inference | `HF_TOKEN` | 60 |
| Llama-2-7b | HuggingFace Inference | `HF_TOKEN` | 30 |

## 타이틀 블록 할당 방식

감정별로 타이틀을 모델에 가중치 비율로 미리 분배합니다.
각 모델은 담당 타이틀에 대해 페르소나를 순환하며 생성합니다.

```
슬픔 (933개 생성, 150 타이틀):
  gpt4o-mini  (0.35) → 타이틀 0~52   → 63회 호출
  gemini      (0.25) → 타이틀 53~89  → 45회 호출
  mistral     (0.20) → 타이틀 90~119 → 36회 호출
  llama2      (0.15) → 타이틀 120~141→ 27회 호출
  gemma       (0.05) → 타이틀 142~149→  9회 호출
```

호출 수 계산:
```
calls_per_title = ceil(per_title / comments_per_prompt)
  예) per_title=6, comments_per_prompt=5 → ceil(6/5) = 2회 호출
```

## 감정별 모델 가중치

`augmentation_config.py`의 `emotion_model_weights`에서 감정별로
어떤 모델이 얼마나 담당할지 가중치를 설정합니다:

```python
"슬픔": {"gpt4o-mini": 0.35, "gemini": 0.25, "mistral": 0.20, ...}
"기쁨": {"llama2": 0.35, "mistral": 0.25, "gemma": 0.20, ...}
```

비활성화된 모델의 가중치는 자동으로 나머지 모델에 재분배됩니다.

## 페르소나

4개 페르소나가 타이틀 순회 시 자동으로 순환됩니다:

| 페르소나 | 설명 |
|----------|------|
| 직장인 | 30대 회사원, 야근이 많고 스트레스를 받는 직장인 |
| 대학생 | 20대 대학생, 취업 준비와 학업에 쫓기는 학생 |
| 주부 | 40대 전업주부, 가정과 육아에 헌신적인 어머니 |
| 은퇴자 | 60대 은퇴자, 사회 변화에 관심이 많은 시니어 |

## 품질 관리

- **길이 제한**: 15~60자
- **한국어 검증**: 최소 3자 한국어 포함
- **메타 응답 필터링**: "네, 알겠습니다" 등 LLM 메타 응답 자동 제거
- **욕설 필터링**: 비속어/과도한 분노 표현 차단
- **중복 제거**: SequenceMatcher 유사도 0.85 이상 중복 제거

## 체크포인트

감정별 생성 완료 시 자동으로 CSV 체크포인트를 저장합니다:

```
augmentation_checkpoints/
├── checkpoint_933.csv
├── checkpoint_1927.csv
└── ...
```

## 평가 리포트

생성 완료 후 자동으로 출력됩니다:

```
[1] 감정별 생성 결과 (기존/생성/합계/목표)
[2] 생성 댓글 길이 분포 (평균, 중앙값, 표준편차)
[3] 어휘 다양성 (TTR)
[4] 모델별 통계 (생성 수, 평균 길이, TTR)
[5] 페르소나별 통계
```

## 데이터 소스

- **기존 감정 데이터**: `MindCastSogang/MindCastTrainSet` (sentiment_comments parquet)
- **뉴스 타이틀**: `MindCastSogang/Youtube_news_preprocessed_data` (preprocessed/v1 JSON)

## 주요 설정 (`augmentation_config.py`)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `anchor_emotion` | `"분노"` | 기준 감정 (이 감정의 개수에 맞춤) |
| `generate_emotions` | 슬픔,불안,상처,당황,기쁨 | 생성 대상 감정 |
| `comments_per_prompt` | 5 | API 1회 호출당 생성할 댓글 수 |
| `min_comment_length` | 15 | 최소 댓글 길이 (자) |
| `max_comment_length` | 60 | 최대 댓글 길이 (자) |
| `duplicate_threshold` | 0.85 | 중복 판단 유사도 기준 |
| `checkpoint_dir` | `augmentation_checkpoints` | 체크포인트 저장 경로 |
