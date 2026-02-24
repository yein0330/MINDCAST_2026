"""데이터셋 모듈 — FullLCombinationDataset

emotion_cols를 지정하면 Stage 1 감정 supervision용 레이블도 반환:
    반환: (X, mask, y_suicide, y_emotion)
    미지정 시: (X, mask, y_suicide)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

EMOTION_COLS_DEFAULT = [
    "감정_분노", "감정_슬픔", "감정_불안", "감정_상처", "감정_당황", "감정_기쁨"
]


class FullLCombinationDataset(Dataset):
    """모든 (L, t) 조합을 전개하는 시계열 데이터셋

    Args:
        df            : 월별 DataFrame (날짜 정렬)
        feature_cols  : Stage 1 입력으로 사용할 사회경제적 지표 컬럼 목록
        emotion_cols  : Stage 1 학습 타겟인 월별 감정 분포 컬럼 목록
                        None이면 감정 supervision 미사용 (suicide loss만)
        pred_offset   : 몇 달 뒤 자살자 수를 예측할지 (기본 4)
    """

    def __init__(
        self,
        df,
        feature_cols,
        emotion_cols=None,
        target_cols=None,
        max_seq_len=32,
        L_min=1,
        L_max=None,
        pred_offset=4,
    ):
        """
        Args:
            target_cols : 예측 타겟 컬럼 리스트.
                          None이면 자동 탐지 (["남자","여자"] → ["자살자수"] 순서로 시도).
                          예: ["자살자수"]  또는  ["남자", "여자"]
        """
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.emotion_cols = emotion_cols  # None이면 감정 supervision 없음

        # 타겟 컬럼 자동 탐지
        if target_cols is None:
            for candidates in [["남자", "여자"], ["자살자수"], ["자살사망자수"]]:
                if all(c in df.columns for c in candidates):
                    target_cols = candidates
                    break
            if target_cols is None:
                raise ValueError(
                    f"타겟 컬럼을 자동 탐지할 수 없습니다. 컬럼 목록: {list(df.columns)}"
                )
        self.target_cols = target_cols
        self.max_len, self.pred_offset = max_seq_len, pred_offset

        # 사회경제적 지표 정규화 통계 (z-score)
        self.X_mu = self.df[self.feature_cols].mean()
        self.X_std = self.df[self.feature_cols].std().replace(0, 1.0)

        self.T = len(df)
        if L_max is None:
            L_max = self.T - pred_offset
        self.samples = [
            (L, t)
            for L in range(L_min, L_max + 1)
            for t in range(L - 1, self.T - pred_offset)
        ]
        mode = "감정 supervision ON" if emotion_cols else "감정 supervision OFF"
        print(f"Generated {len(self.samples)} samples (L={L_min}~{L_max}) [{mode}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        L, t = self.samples[idx]
        start = t - L + 1

        # ---- 입력 시퀀스 (사회경제 지표, z-score 정규화) ----
        X = (self.df.loc[start:t, self.feature_cols] - self.X_mu) / self.X_std
        D = len(self.feature_cols)
        pad_len = self.max_len - L
        pad = np.zeros((pad_len, D), np.float32)
        Xv = np.concatenate([pad, X.values.astype(np.float32)], 0)
        mask = np.array([True] * pad_len + [False] * L, dtype=np.bool_)

        # ---- Stage 2 타겟: t+offset 시점의 자살자 수 ----
        y_suicide = self.df.loc[
            t + self.pred_offset, self.target_cols
        ].values.astype(np.float32)

        if self.emotion_cols is not None:
            # ---- Stage 1 타겟: t 시점의 감정 분포 (비율, 0~1) ----
            y_emotion = self.df.loc[t, self.emotion_cols].values.astype(np.float32)
            return (
                torch.from_numpy(Xv).float(),
                torch.from_numpy(mask),
                torch.from_numpy(y_suicide).float(),
                torch.from_numpy(y_emotion).float(),
            )

        return (
            torch.from_numpy(Xv).float(),
            torch.from_numpy(mask),
            torch.from_numpy(y_suicide).float(),
        )
