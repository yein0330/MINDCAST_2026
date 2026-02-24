"""데이터 로더 — HuggingFace Private 데이터셋에서 학습 데이터 구성

1. MindCastSogang/SuicideDataset
       → hf_hub_download()로 루트 CSV 직접 로드 → 월별 사회경제 지표 + 자살 통계
       → (load_dataset() 불가: 레포 내 파일들 컬럼 스키마 불일치)

2. YenYein/Youtube_comments_anaylsis_ver1
       → 파일 구조: {year}/{month}/{date_range}/infer_*.json
       → 각 JSON 내부:
           {
             "data": [{
               "date": "YYYY-MM-DD",
               "posts": [{
                 "news_date": "YYYY-MM-DD",
                 "analyses": {
                   "SentimentClassificationPipeLine_comments": [
                     [{"label": "분노", "score": 0.54}],  # 댓글 1
                     [{"label": "슬픔", "score": 0.43}],  # 댓글 2
                     ...
                   ]
                 }
               }]
             }]
           }
       → 모든 infer_*.json을 순회하여 댓글 감정 레이블 수집
       → news_date 기준 월별 집계 → 감정 비율 (0~1)

3. 두 데이터를 날짜 기준 inner join → base_data.csv 저장

사용법:
    # 댓글 데이터 파일 목록만 확인
    python -m emotion_suicide_trainer.data_loader --inspect --token YOUR_HF_TOKEN

    # 전체 전처리 실행
    python -m emotion_suicide_trainer.data_loader --token YOUR_HF_TOKEN \\
        --output /home/mindcastlib/data/base/base_data.csv
"""
import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files, login

# ============================================================
# 상수
# ============================================================
SUICIDE_REPO = "MindCastSogang/SuicideDataset"
COMMENT_REPO = "YenYein/Youtube_comments_anaylsis_ver1"

EMOTION_NAMES = ["분노", "슬픔", "불안", "상처", "당황", "기쁨"]
EMOTION_COLS = [f"감정_{e}" for e in EMOTION_NAMES]

# 댓글 감정 분류 결과가 들어있는 키
SENTIMENT_KEY = "SentimentClassificationPipeLine_comments"


# ============================================================
# Step 0: 데이터셋 구조 확인
# ============================================================
def inspect_datasets(token: str):
    """두 데이터셋 구조 출력"""

    # ----- SuicideDataset -----
    print("=" * 60)
    print(f"[INSPECT] {SUICIDE_REPO}")
    print("=" * 60)
    try:
        all_files = list(list_repo_files(SUICIDE_REPO, repo_type="dataset", token=token))
        print(f"\n  전체 파일 목록 ({len(all_files)}개):")
        for f in all_files:
            print(f"    {f}")

        root_csvs = [f for f in all_files if "/" not in f and f.endswith(".csv")]
        base_csvs = sorted(
            [f for f in root_csvs if "base_data" in f or "basedata" in f],
            reverse=True,
        )
        sample_files = base_csvs[:2] if base_csvs else root_csvs[:2]
        for rel_path in sample_files:
            local = hf_hub_download(
                SUICIDE_REPO, rel_path, repo_type="dataset", token=token
            )
            df = pd.read_csv(local)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            print(f"\n  파일: {rel_path}  rows={len(df)}")
            print(f"  columns: {list(df.columns)}")
            print(f"  dtypes:\n{df.dtypes.to_string()}")
            print(f"\n  sample (3 rows):")
            print(df.head(3).to_string())
    except Exception as e:
        print(f"  ERROR: {e}")

    # ----- Comment Repo: 파일 목록 + 샘플 JSON -----
    print("\n" + "=" * 60)
    print(f"[INSPECT] {COMMENT_REPO}")
    print("=" * 60)
    try:
        files = _list_infer_files(token)
        print(f"\n  infer_*.json 파일 수: {len(files)}")
        print(f"  예시 (앞 10개):")
        for f in files[:10]:
            print(f"    {f}")

        if files:
            print(f"\n  첫 번째 파일 샘플 파싱: {files[0]}")
            local = hf_hub_download(
                COMMENT_REPO, files[0], repo_type="dataset", token=token
            )
            with open(local, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            entries = obj.get("data", [])
            print(f"  data 엔트리 수: {len(entries)}")
            if entries:
                e = entries[0]
                print(f"  entry[0].date: {e.get('date')}")
                posts = e.get("posts", [])
                print(f"  entry[0] post 수: {len(posts)}")
                if posts:
                    p = posts[0]
                    print(f"  post[0].news_date: {p.get('news_date')}")
                    sa = p.get("analyses", {}).get(SENTIMENT_KEY, [])
                    print(f"  SentimentClassification 댓글 수: {len(sa)}")
                    if sa:
                        print(f"  댓글[0] 결과: {sa[0]}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ============================================================
# Step 1: 자살 데이터셋 로드
# ============================================================
def load_suicide_dataset(token: str) -> pd.DataFrame:
    """
    MindCastSogang/SuicideDataset 로드 → 월별 DataFrame

    load_dataset() 대신 hf_hub_download로 CSV를 직접 읽음.
    (레포 내 여러 CSV가 컬럼 스키마가 달라 load_dataset()이 실패하므로)

    우선순위:
        1) 루트에 있는 *base_data*.csv / *basedata*.csv (최신 날짜순)
        2) 없으면 루트 *.csv 전체

    반환 컬럼:
        날짜             : YYYY-MM-DD (월 첫날)
        자살자수          : 전체 자살자 수 (혹은 남자/여자 분리 컬럼)
        [기타 사회경제적 지표 컬럼들]
    """
    print(f"[LOAD] {SUICIDE_REPO} ...")

    # 레포 파일 목록 조회
    all_files = list(list_repo_files(SUICIDE_REPO, repo_type="dataset", token=token))

    # 루트 레벨 CSV만 (하위 디렉토리 제외)
    root_csvs = [f for f in all_files if "/" not in f and f.endswith(".csv")]

    # base_data 파일 우선 (날짜 내림차순 → 최신 파일 우선)
    base_csvs = sorted(
        [f for f in root_csvs if "base_data" in f or "basedata" in f],
        reverse=True,
    )
    target_files = base_csvs if base_csvs else sorted(root_csvs, reverse=True)

    if not target_files:
        raise FileNotFoundError(
            f"{SUICIDE_REPO} 에서 로드할 CSV를 찾지 못했습니다.\n"
            f"  전체 파일 목록: {all_files}"
        )

    print(f"  대상 파일: {target_files}")

    # 각 CSV 개별 다운로드 + 읽기
    dfs = []
    for rel_path in target_files:
        local = hf_hub_download(
            SUICIDE_REPO, rel_path, repo_type="dataset", token=token
        )
        _df = pd.read_csv(local)
        # Unnamed 인덱스 컬럼 제거
        _df = _df.loc[:, ~_df.columns.str.contains("^Unnamed")]
        print(f"  {rel_path}: {len(_df)} rows, cols={list(_df.columns)}")
        dfs.append(_df)

    # 컬럼이 동일한 파일끼리만 concat (다르면 가장 컬럼 많은 파일 사용)
    if len(dfs) == 1:
        df = dfs[0]
    else:
        col_sets = [set(d.columns) for d in dfs]
        if all(c == col_sets[0] for c in col_sets):
            df = pd.concat(dfs, ignore_index=True)
        else:
            # 컬럼이 다른 경우 → 가장 많은 컬럼을 가진 파일 선택
            df = max(dfs, key=lambda d: len(d.columns))
            print(f"  WARNING: 파일 간 컬럼 불일치 → 컬럼 최다 파일만 사용 ({len(df.columns)}개 컬럼)")

    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # ---- 날짜 정규화 ('YYYY-MM' 또는 'YYYY-MM-DD' 모두 처리) ----
    date_col = _detect_date_col(df)
    df = df.rename(columns={date_col: "날짜"})
    df["날짜"] = (
        pd.to_datetime(df["날짜"].astype(str), format="mixed", dayfirst=False)
        .dt.to_period("M")
        .dt.to_timestamp()          # 월 첫날 00:00:00
        .dt.strftime("%Y-%m-%d")
    )

    # ---- 중복 월 제거 (같은 달이 두 행 이상이면 평균) ----
    dupes = df.groupby("날짜").size()
    if (dupes > 1).any():
        dup_months = dupes[dupes > 1].index.tolist()
        print(f"  WARNING: 중복 월 발견 ({len(dup_months)}개) → 평균으로 합산")
        num_cols = df.select_dtypes("number").columns.tolist()
        df = df.groupby("날짜")[num_cols].mean().reset_index()

    df = df.sort_values("날짜").reset_index(drop=True)
    print(f"  Date range: {df['날짜'].iloc[0]} ~ {df['날짜'].iloc[-1]}")

    # ---- 타겟 컬럼 존재 여부 안내 ----
    for col in ["남자", "여자", "자살자수", "자살사망자수"]:
        if col in df.columns:
            print(f"  타겟 컬럼 확인: '{col}' ✓")
    return df


# ============================================================
# Step 2: 댓글 감정 집계 — JSON 파일 직접 파싱
# ============================================================
def load_and_aggregate_emotion(token: str) -> pd.DataFrame:
    """
    YenYein/Youtube_comments_anaylsis_ver1 의 모든 infer_*.json을 순회하여
    댓글 감정 레이블을 news_date 기준 월별로 집계 → 비율 반환

    파일은 댓글 생성일 기준으로 분류되어 있어 동일 뉴스 기사(title)가
    여러 파일에 걸쳐 중복 등장할 수 있으므로, unique 뉴스 수는 set으로 dedup.

    반환 컬럼:
        날짜         : YYYY-MM-DD (월 첫날)
        감정_분노     : 해당 월 분노 댓글 비율 (0~1)
        감정_슬픔, 감정_불안, 감정_상처, 감정_당황, 감정_기쁨
        댓글수        : 해당 월 총 댓글 수 (감정 분류된 댓글)
        뉴스수        : 해당 월 unique 뉴스 기사 수 (title 기준 dedup)
    """
    files = _list_infer_files(token)
    print(f"[LOAD] {COMMENT_REPO} — infer_*.json 파일 {len(files)}개 처리 시작 ...")

    # {YYYY-MM: {감정: count}}
    monthly_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # {YYYY-MM: 총 댓글 수}
    monthly_comment_count: dict[str, int] = defaultdict(int)
    # {YYYY-MM: set of unique 뉴스 식별자}
    monthly_unique_posts: dict[str, set] = defaultdict(set)
    total_files = len(files)

    for i, rel_path in enumerate(files, 1):
        if i % 50 == 0 or i == total_files:
            print(f"  [{i}/{total_files}] {rel_path}")

        try:
            local = hf_hub_download(
                COMMENT_REPO, rel_path, repo_type="dataset", token=token
            )
            with open(local, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception as e:
            print(f"  SKIP {rel_path}: {e}")
            continue

        _parse_and_accumulate(obj, monthly_counts, monthly_comment_count, monthly_unique_posts)

    return _build_emotion_df(monthly_counts, monthly_comment_count, monthly_unique_posts)


def _list_infer_files(token: str) -> list[str]:
    """레포 내 모든 infer_*.json 경로를 반환 (패턴: {year}/{month}/*/infer_*.json)"""
    all_files = list(list_repo_files(COMMENT_REPO, repo_type="dataset", token=token))
    pattern = re.compile(r"^\d{4}/\d{2}/.+/infer_.+\.json$")
    return sorted(f for f in all_files if pattern.match(f))


def _parse_and_accumulate(
    obj: dict,
    monthly_counts: dict,
    monthly_comment_count: dict,
    monthly_unique_posts: dict,
):
    """
    단일 JSON 파일을 파싱하여 monthly_* 누적 딕셔너리에 추가.

    - monthly_counts      : {YYYY-MM: {감정: count}}
    - monthly_comment_count: {YYYY-MM: 총 댓글 수}
    - monthly_unique_posts : {YYYY-MM: set(뉴스 식별자)}
        뉴스 식별자 = title / 제목 / url 중 존재하는 첫 필드.
        없으면 (news_date, post 순서 index) 로 fallback.

    파일은 댓글 생성일 기준이므로 동일 뉴스가 여러 파일에 중복 등장 가능.
    unique_posts set이 파일 간 중복을 자동 제거함.
    """
    for entry in obj.get("data", []):
        fallback_date = entry.get("date", "")
        for post_idx, post in enumerate(entry.get("posts", [])):
            # news_date 우선, 없으면 entry.date 사용
            raw_date = post.get("news_date") or fallback_date
            if not raw_date:
                continue
            ym = _to_year_month(raw_date)  # "YYYY-MM"
            if ym is None:
                continue

            # ---- unique 뉴스 식별자 ----
            post_id = (
                post.get("title")
                or post.get("제목")
                or post.get("url")
                or post.get("link")
                or f"{raw_date}__{post_idx}"  # fallback
            )
            monthly_unique_posts[ym].add(post_id)

            # ---- 댓글 감정 집계 ----
            sent_pipeline = post.get("analyses", {}).get(SENTIMENT_KEY, [])
            for comment_results in sent_pipeline:
                # comment_results: [{"label": "분노", "score": 0.54}]
                if not comment_results:
                    continue
                label = comment_results[0].get("label", "")
                if label in EMOTION_NAMES:
                    monthly_counts[ym][label] += 1
                    monthly_comment_count[ym] += 1


def _to_year_month(date_str: str):
    """'YYYY-MM-DD' 또는 'YYYY-MM' → 'YYYY-MM'. 파싱 실패 시 None."""
    try:
        return pd.to_datetime(date_str).strftime("%Y-%m")
    except Exception:
        return None


def _build_emotion_df(
    monthly_counts: dict,
    monthly_comment_count: dict,
    monthly_unique_posts: dict,
) -> pd.DataFrame:
    """monthly_* → 날짜 정렬된 감정 비율 + 댓글수 + 뉴스수 DataFrame"""
    all_months = sorted(
        set(monthly_counts.keys())
        | set(monthly_comment_count.keys())
        | set(monthly_unique_posts.keys())
    )
    rows = []
    for ym in all_months:
        counts = monthly_counts[ym]
        total = sum(counts.values())
        row = {"날짜": f"{ym}-01"}
        for emo in EMOTION_NAMES:
            row[f"감정_{emo}"] = counts.get(emo, 0) / total if total > 0 else 0.0
        row["댓글수"] = monthly_comment_count.get(ym, 0)
        row["뉴스수"] = len(monthly_unique_posts.get(ym, set()))
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Aggregated to {len(df)} monthly rows")
    if len(df):
        print(f"  Date range: {df['날짜'].iloc[0]} ~ {df['날짜'].iloc[-1]}")
        print(f"  댓글수 범위: {df['댓글수'].min():.0f} ~ {df['댓글수'].max():.0f}")
        print(f"  뉴스수 범위: {df['뉴스수'].min():.0f} ~ {df['뉴스수'].max():.0f}")
    return df


# ============================================================
# Step 3: 병합 → base_data.csv
# ============================================================
def build_base_data(token: str, output_path: str = "base_data.csv") -> pd.DataFrame:
    """
    두 데이터셋을 로드·병합하여 base_data.csv 저장

    병합 전략:
        - 자살 데이터를 기준(left)으로 left join
        - 댓글 감정 데이터가 없는 월은 감정 컬럼을 0.0으로 채움
        - 두 데이터의 날짜 범위가 달라도 자살 데이터 전체 보존

    최종 컬럼 순서:
        날짜 | [사회경제지표...] | 감정_분노~기쁨 | 남자 | 여자 | 자살사망자수
    """
    df_suicide = load_suicide_dataset(token)
    df_emotion = load_and_aggregate_emotion(token)

    # ---- 날짜 범위 커버리지 리포트 ----
    suicide_months = set(df_suicide["날짜"])
    emotion_months = set(df_emotion["날짜"]) if len(df_emotion) else set()
    overlap = suicide_months & emotion_months
    only_suicide = suicide_months - emotion_months
    only_emotion = emotion_months - suicide_months
    print(f"\n[DATE COVERAGE]")
    print(f"  자살 데이터 : {len(suicide_months)}개월 "
          f"({sorted(suicide_months)[0]} ~ {sorted(suicide_months)[-1]})")
    if emotion_months:
        print(f"  댓글 감정   : {len(emotion_months)}개월 "
              f"({sorted(emotion_months)[0]} ~ {sorted(emotion_months)[-1]})")
    print(f"  공통 구간   : {len(overlap)}개월")
    if only_suicide:
        print(f"  감정 없는 월 (0 채움): {len(only_suicide)}개월")
    if only_emotion:
        print(f"  자살 없는 월 (제외): {len(only_emotion)}개월")

    # ---- Left join: 자살 데이터 기준 ----
    df_merged = pd.merge(df_suicide, df_emotion, on="날짜", how="left")
    print(f"\n[MERGE] {len(df_merged)} months after left join (suicide data as primary)")

    # 댓글 기반 컬럼 NaN 채움 (댓글 데이터 없는 월)
    fill_zero_cols = EMOTION_COLS + ["댓글수", "뉴스수"]
    for col in fill_zero_cols:
        if col in df_merged.columns:
            n_filled = df_merged[col].isna().sum()
            if n_filled:
                print(f"  {col}: {n_filled}개월 NaN → 0 채움")
            df_merged[col] = df_merged[col].fillna(0)
        else:
            df_merged[col] = 0
            print(f"  {col}: 컬럼 없음 → 0 으로 생성")

    for col in ["남자", "여자", "자살사망자수"]:
        if col not in df_merged.columns:
            print(f"  WARNING: '{col}' 컬럼 없음 — 실제 컬럼명 확인 필요")

    df_merged = df_merged.sort_values("날짜").reset_index(drop=True)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    df_merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] → {output_path}")
    print(f"  rows={len(df_merged)}, cols={list(df_merged.columns)}")
    return df_merged


# ============================================================
# 유틸: SuicideDataset 날짜 컬럼 자동 탐지
# ============================================================
def _detect_date_col(df: pd.DataFrame) -> str:
    candidates = ["날짜", "date", "Date", "DATE", "year_month",
                  "month", "기준년월", "기준일", "연도월", "ym"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    raise ValueError(
        f"날짜 컬럼을 자동 탐지할 수 없습니다. 컬럼 목록: {list(df.columns)}\n"
        "_detect_date_col()의 candidates에 실제 컬럼명을 추가하세요."
    )


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="HuggingFace 데이터 로더")
    parser.add_argument("--token", required=True, help="HuggingFace 액세스 토큰")
    parser.add_argument("--inspect", action="store_true",
                        help="데이터셋 구조만 출력 (전처리 없음)")
    parser.add_argument("--output", default="/home/yein38/mindcastlib_trainer/data/base/base_data.csv",
                        help="저장할 CSV 경로")
    args = parser.parse_args()

    login(token=args.token)

    if args.inspect:
        inspect_datasets(args.token)
    else:
        build_base_data(args.token, args.output)


if __name__ == "__main__":
    main()
