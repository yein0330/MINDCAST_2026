"""Supervised emotion -> suicide 2-stage pipeline.

목표:
  Stage1: 사회/경제 스트레스 시계열 -> 실제 감정 분포 예측 (supervised)
  Stage2: 감정 분포 -> 자살자 수 예측
  Joint : Stage1 예측 감정 분포를 Stage2에 넣고 joint fine-tuning

실제 감정 분포 데이터 컬럼명이 확정되면 `main()`의 `emotion_cols`만 지정해서 사용 가능.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .model import EmotionAmplifier, SuicideForecaster, EmotionSequenceForecaster
from .utils import (
    CSV_PATH,
    SAVE_ROOT,
    MAX_LEN,
    BATCH_SIZE,
    EPOCHS,
    LR,
    SEED,
    set_seed,
    load_clean_csv,
)

EMOTION_EN = {
    "감정_분노": "Anger",
    "감정_슬픔": "Sadness",
    "감정_불안": "Anxiety",
    "감정_상처": "Hurt",
    "감정_당황": "Embarrassment",
    "감정_기쁨": "Joy",
}
TARGET_EN = {
    "자살자수": "Suicide Count",
    "자살사망자수": "Suicide Deaths",
    "남자": "Male",
    "여자": "Female",
}


def _en(col: str) -> str:
    """Return English display name for a Korean column."""
    return EMOTION_EN.get(col, TARGET_EN.get(col, col.replace("감정_", "")))


@dataclass
class PipelineConfig:
    csv_path: str = CSV_PATH
    save_root: str = os.path.join(SAVE_ROOT, "supervised_pipeline")
    max_seq_len: int = MAX_LEN
    batch_size: int = BATCH_SIZE
    stage1_epochs: int = 50
    stage2_epochs: int = 50
    joint_epochs: int = 50
    lr_stage1: float = LR
    lr_stage2: float = LR
    lr_joint: float = LR * 0.5
    pred_offset: int = 4
    seed: int = SEED
    d_model: int = 128
    n_heads: int = 4
    d_hidden: int = 64
    dropout: float = 0.1
    lambda_emotion: float = 1.0
    lambda_suicide: float = 1.0
    normalize_emotion_target: bool = True
    # "mse" | "kl"
    emotion_loss_type: str = "mse"
    # log1p-transform suicide targets to fix scale mismatch
    log_transform_suicide: bool = True
    # Stage 2: number of past emotion months to attend over (lag window size)
    emotion_lag: int = 6


class EmotionSuicideDataset(Dataset):
    """감정 분포 + 자살자 수를 함께 반환하는 시계열 데이터셋."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        emotion_cols: Sequence[str],
        target_cols: Optional[Sequence[str]] = None,
        max_seq_len: int = 32,
        L_min: int = 1,
        L_max: Optional[int] = None,
        pred_offset: int = 4,
        normalize_emotion_target: bool = True,
        log_transform_suicide: bool = True,
        emotion_lag: int = 6,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = list(feature_cols)
        self.emotion_cols = list(emotion_cols)
        self.max_len = max_seq_len
        self.pred_offset = pred_offset
        self.normalize_emotion_target = normalize_emotion_target
        self.log_transform_suicide = log_transform_suicide
        self.emotion_lag = emotion_lag

        # 타겟 컬럼 자동 탐지 (남자+여자 > 자살자수 > 자살사망자수)
        if target_cols is None:
            for candidates in [["남자", "여자"], ["자살자수"], ["자살사망자수"]]:
                if all(c in df.columns for c in candidates):
                    target_cols = candidates
                    break
            if target_cols is None:
                raise ValueError(f"타겟 컬럼을 자동 탐지할 수 없습니다. 컬럼 목록: {list(df.columns)}")
        self.target_cols = list(target_cols)

        missing = [c for c in self.feature_cols + self.emotion_cols + self.target_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.X_mu = self.df[self.feature_cols].mean()
        self.X_std = self.df[self.feature_cols].std().replace(0, 1.0)
        self.T = len(self.df)
        if L_max is None:
            L_max = self.T - pred_offset
        self.samples = [
            (L, t)
            for L in range(L_min, L_max + 1)
            for t in range(L - 1, self.T - pred_offset)
        ]
        print(f"[EmotionSuicideDataset] Generated {len(self.samples)} samples (L={L_min}~{L_max})")
        print(f"  target_cols: {self.target_cols}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        L, t = self.samples[idx]
        start = t - L + 1

        X = (self.df.loc[start:t, self.feature_cols] - self.X_mu) / self.X_std
        D = len(self.feature_cols)
        pad_len = self.max_len - L
        pad = np.zeros((pad_len, D), np.float32)
        Xv = np.concatenate([pad, X.values.astype(np.float32)], axis=0)
        mask = np.array([True] * pad_len + [False] * L, dtype=np.bool_)

        # y_suicide: future target (t + offset)
        y_suicide = self.df.loc[t + self.pred_offset, self.target_cols].values.astype(np.float32)
        if self.log_transform_suicide:
            y_suicide = np.log1p(y_suicide)

        # Emotion sequence: [t-lag, ..., t]  (Stage 1 target = last step = emotion[t])
        emo_end = t
        emo_start = max(0, t - self.emotion_lag)
        emo_raw = self.df.loc[emo_start:emo_end, self.emotion_cols].values.astype(np.float32)
        seq_len = emo_end - emo_start + 1
        emo_pad_len = self.emotion_lag + 1 - seq_len

        if self.normalize_emotion_target:
            row_sums = emo_raw.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            emo_raw = emo_raw / row_sums

        emo_pad = np.zeros((emo_pad_len, len(self.emotion_cols)), np.float32)
        emo_seq = np.concatenate([emo_pad, emo_raw], axis=0)   # (lag+1, n_emotions)
        emo_mask = np.array([True] * emo_pad_len + [False] * seq_len, dtype=np.bool_)

        return (
            torch.from_numpy(Xv).float(),              # (max_len, d_in)
            torch.from_numpy(mask),                    # (max_len,) stress mask
            torch.from_numpy(emo_seq).float(),         # (lag+1, n_emotions)
            torch.from_numpy(emo_mask),                # (lag+1,) emotion mask
            torch.from_numpy(y_suicide).float(),       # (n_out,)
        )


class StressToEmotionModel(nn.Module):
    """Stage1: 스트레스 시계열 -> 감정 분포."""

    def __init__(
        self,
        d_in: int,
        n_emotions: int,
        d_model: int = 128,
        n_heads: int = 4,
        max_len: int = 32,
        dropout: float = 0.1,
        output_distribution: bool = True,
    ):
        super().__init__()
        self.encoder = EmotionAmplifier(
            d_in=d_in,
            d_model=d_model,
            n_heads=n_heads,
            n_emotions=n_emotions,
            max_len=max_len,
            dropout=dropout,
        )
        self.output_distribution = output_distribution

    def forward(self, X, mask, return_attn: bool = False):
        if return_attn:
            logits, attn = self.encoder(X, mask, return_attn=True)
        else:
            logits = self.encoder(X, mask)
        pred = torch.softmax(logits, dim=-1) if self.output_distribution else logits
        if return_attn:
            return pred, attn
        return pred


class EmotionToSuicideModel(nn.Module):
    """Stage 2: emotion sequence (lag window) → suicide count.

    Stage 1과 동일한 Cross-Attention 구조:
      input  : (B, lag+1, n_emotions) — 여러 시점의 감정 분포
      output : (B, n_out)             — 자살자 수 예측
    """

    def __init__(
        self,
        n_emotions: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_out: int = 1,
        emotion_lag: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.forecaster = EmotionSequenceForecaster(
            n_emotions=n_emotions,
            d_model=d_model,
            n_heads=n_heads,
            n_out=n_out,
            max_lag=emotion_lag + 1,
            dropout=dropout,
        )

    def forward(self, emotion_seq, mask=None, return_attn: bool = False):
        return self.forecaster(emotion_seq, mask, return_attn=return_attn)


class JointEmotionSuicideModel(nn.Module):
    """Joint model: Stage 1 predicts current emotion, Stage 2 attends over lag sequence.

    Stage 2 입력 = [GT_emotion[t-lag..t-1], predicted_emotion[t]]
    → 과거 감정은 GT, 현재 감정은 Stage 1 예측을 사용하여 end-to-end 학습
    """

    def __init__(self, stage1: StressToEmotionModel, stage2: EmotionToSuicideModel):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2

    def forward(self, X, stress_mask, emo_hist, emo_hist_mask):
        """
        Args:
            X             : (B, T, d_in)       stress sequence
            stress_mask   : (B, T)             True = padding
            emo_hist      : (B, lag, n_emotions) GT emotions for [t-lag .. t-1]
            emo_hist_mask : (B, lag)           True = padding
        Returns:
            pred_emotion : (B, n_emotions)
            pred_suicide : (B, n_out)
        """
        pred_emotion = self.stage1(X, stress_mask)          # (B, n_emotions)

        # Append predicted current emotion to historical sequence
        pred_cur = pred_emotion.unsqueeze(1)                # (B, 1, n_emotions)
        full_emo_seq = torch.cat([emo_hist, pred_cur], dim=1)   # (B, lag+1, n_emotions)

        # Current step is never padding
        cur_not_pad = torch.zeros(
            emo_hist_mask.size(0), 1, dtype=torch.bool, device=emo_hist_mask.device
        )
        full_emo_mask = torch.cat([emo_hist_mask, cur_not_pad], dim=1)  # (B, lag+1)

        pred_suicide = self.stage2(full_emo_seq, full_emo_mask)
        return pred_emotion, pred_suicide


def _emotion_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    if loss_type == "mse":
        return nn.functional.mse_loss(pred, target)
    if loss_type == "kl":
        eps = 1e-8
        pred_log = torch.log(pred.clamp_min(eps))
        target_prob = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
        return nn.functional.kl_div(pred_log, target_prob, reduction="batchmean")
    raise ValueError(f"Unknown emotion loss type: {loss_type}")


@torch.no_grad()
def evaluate_stage1(model, loader, device, emotion_loss_type="mse"):
    model.eval()
    emo_sum = 0.0
    for X, stress_mask, emo_seq, emo_mask, _ in loader:
        X = X.to(device)
        stress_mask = stress_mask.to(device)
        y_emotion = emo_seq[:, -1, :].to(device)   # emotion at current time t
        pred = model(X, stress_mask)
        emo_sum += _emotion_loss(pred, y_emotion, emotion_loss_type).item() * X.size(0)
    return {"emotion_loss": emo_sum / len(loader.dataset)}


@torch.no_grad()
def evaluate_stage2(model, loader, device):
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total = 0.0
    for _, _, emo_seq, emo_mask, y_suicide in loader:
        emo_seq = emo_seq.to(device)
        emo_mask = emo_mask.to(device)
        y_suicide = y_suicide.to(device)
        pred = model(emo_seq, emo_mask)
        total += mse(pred, y_suicide).item()
    return {"suicide_mse": total / len(loader.dataset)}


@torch.no_grad()
def evaluate_joint(model, loader, device, emotion_loss_type="mse", lam_e=1.0, lam_s=1.0):
    model.eval()
    mse_sum = 0.0
    emo_sum = 0.0
    for X, stress_mask, emo_seq, emo_mask, y_suicide in loader:
        X = X.to(device)
        stress_mask = stress_mask.to(device)
        y_emotion = emo_seq[:, -1, :].to(device)
        emo_hist = emo_seq[:, :-1, :].to(device)
        emo_hist_mask = emo_mask[:, :-1].to(device)
        y_suicide = y_suicide.to(device)
        pred_emotion, pred_suicide = model(X, stress_mask, emo_hist, emo_hist_mask)
        emo = _emotion_loss(pred_emotion, y_emotion, emotion_loss_type)
        suc = nn.functional.mse_loss(pred_suicide, y_suicide)
        emo_sum += emo.item() * X.size(0)
        mse_sum += suc.item() * X.size(0)
    emo_avg = emo_sum / len(loader.dataset)
    mse_avg = mse_sum / len(loader.dataset)
    return {
        "emotion_loss": emo_avg,
        "suicide_mse": mse_avg,
        "total_loss": lam_e * emo_avg + lam_s * mse_avg,
    }


def train_stage1(model, train_loader, val_loader, cfg: PipelineConfig, device):
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr_stage1)
    best_state = None
    best_val = float("inf")
    history = []

    for ep in range(1, cfg.stage1_epochs + 1):
        model.train()
        running = 0.0
        for X, stress_mask, emo_seq, emo_mask, _ in train_loader:
            X = X.to(device)
            stress_mask = stress_mask.to(device)
            y_emotion = emo_seq[:, -1, :].to(device)   # emotion at current time t
            optim.zero_grad(set_to_none=True)
            pred = model(X, stress_mask)
            loss = _emotion_loss(pred, y_emotion, cfg.emotion_loss_type)
            loss.backward()
            optim.step()
            running += loss.item() * X.size(0)
        tr = running / len(train_loader.dataset)
        va = evaluate_stage1(model, val_loader, device, cfg.emotion_loss_type)["emotion_loss"]
        history.append({"epoch": ep, "train_emotion_loss": tr, "val_emotion_loss": va})
        print(f"[Stage1][{ep:03d}] train_emotion={tr:.6f} val_emotion={va:.6f}")
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_stage2(model, train_loader, val_loader, cfg: PipelineConfig, device):
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr_stage2)
    best_state = None
    best_val = float("inf")
    history = []
    mse = nn.MSELoss()

    for ep in range(1, cfg.stage2_epochs + 1):
        model.train()
        running = 0.0
        for _, _, emo_seq, emo_mask, y_suicide in train_loader:
            emo_seq = emo_seq.to(device)
            emo_mask = emo_mask.to(device)
            y_suicide = y_suicide.to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(emo_seq, emo_mask)
            loss = mse(pred, y_suicide)
            loss.backward()
            optim.step()
            running += loss.item() * emo_seq.size(0)
        tr = running / len(train_loader.dataset)
        va = evaluate_stage2(model, val_loader, device)["suicide_mse"]
        history.append({"epoch": ep, "train_suicide_mse": tr, "val_suicide_mse": va})
        print(f"[Stage2][{ep:03d}] train_suicide={tr:.6f} val_suicide={va:.6f}")
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_joint(model, train_loader, val_loader, cfg: PipelineConfig, device):
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr_joint)
    best_state = None
    best_val = float("inf")
    history = []

    for ep in range(1, cfg.joint_epochs + 1):
        model.train()
        running_total = 0.0
        running_e = 0.0
        running_s = 0.0
        for X, stress_mask, emo_seq, emo_mask, y_suicide in train_loader:
            X = X.to(device)
            stress_mask = stress_mask.to(device)
            y_emotion = emo_seq[:, -1, :].to(device)
            emo_hist = emo_seq[:, :-1, :].to(device)
            emo_hist_mask = emo_mask[:, :-1].to(device)
            y_suicide = y_suicide.to(device)
            optim.zero_grad(set_to_none=True)
            pred_emotion, pred_suicide = model(X, stress_mask, emo_hist, emo_hist_mask)
            l_emotion = _emotion_loss(pred_emotion, y_emotion, cfg.emotion_loss_type)
            l_suicide = nn.functional.mse_loss(pred_suicide, y_suicide)
            loss = cfg.lambda_emotion * l_emotion + cfg.lambda_suicide * l_suicide
            loss.backward()
            optim.step()

            bs = X.size(0)
            running_total += loss.item() * bs
            running_e += l_emotion.item() * bs
            running_s += l_suicide.item() * bs

        tr_total = running_total / len(train_loader.dataset)
        tr_e = running_e / len(train_loader.dataset)
        tr_s = running_s / len(train_loader.dataset)
        va = evaluate_joint(
            model, val_loader, device, cfg.emotion_loss_type, cfg.lambda_emotion, cfg.lambda_suicide
        )
        history.append({
            "epoch": ep,
            "train_total": tr_total,
            "train_emotion_loss": tr_e,
            "train_suicide_mse": tr_s,
            **{f"val_{k}": v for k, v in va.items()},
        })
        print(
            f"[Joint][{ep:03d}] train_total={tr_total:.6f} "
            f"(emotion={tr_e:.6f}, suicide={tr_s:.6f}) "
            f"val_total={va['total_loss']:.6f}"
        )
        if va["total_loss"] < best_val:
            best_val = va["total_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def build_dataloaders(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    emotion_cols: Sequence[str],
    cfg: PipelineConfig,
    target_cols: Optional[Sequence[str]] = None,
):
    dataset = EmotionSuicideDataset(
        df=df,
        feature_cols=feature_cols,
        emotion_cols=emotion_cols,
        target_cols=target_cols,
        max_seq_len=cfg.max_seq_len,
        pred_offset=cfg.pred_offset,
        normalize_emotion_target=cfg.normalize_emotion_target,
        log_transform_suicide=cfg.log_transform_suicide,
        emotion_lag=cfg.emotion_lag,
    )
    # Time-based split: last 20% of unique target timesteps → val.
    # Prevents the same target month appearing in both train and val.
    all_target_times = sorted(set(t + dataset.pred_offset for _, t in dataset.samples))
    n_val_times = max(1, int(len(all_target_times) * 0.2))
    val_cutoff = all_target_times[-n_val_times]
    train_idx = [i for i, (_, t) in enumerate(dataset.samples) if t + dataset.pred_offset < val_cutoff]
    val_idx   = [i for i, (_, t) in enumerate(dataset.samples) if t + dataset.pred_offset >= val_cutoff]
    print(f"  Time split: train targets <{val_cutoff}, val targets >={val_cutoff}")
    print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    return dataset, train_loader, val_loader


def save_histories(save_dir: str, stage1_hist, stage2_hist, joint_hist):
    os.makedirs(save_dir, exist_ok=True)
    if stage1_hist:
        pd.DataFrame(stage1_hist).to_csv(os.path.join(save_dir, "stage1_history.csv"), index=False)
    if stage2_hist:
        pd.DataFrame(stage2_hist).to_csv(os.path.join(save_dir, "stage2_history.csv"), index=False)
    if joint_hist:
        pd.DataFrame(joint_hist).to_csv(os.path.join(save_dir, "joint_history.csv"), index=False)


def plot_results(
    save_dir: str,
    stage1_hist: list,
    stage2_hist: list,
    joint_hist: list,
    joint_model,
    val_loader,
    target_cols: Sequence[str],
    emotion_cols: Sequence[str],
    device: str,
    log_transform_suicide: bool = False,
):
    """Loss curves + pred-vs-actual plots (English labels)."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Loss curves ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if stage1_hist:
        h = pd.DataFrame(stage1_hist)
        axes[0].plot(h["epoch"], h["train_emotion_loss"], label="train")
        axes[0].plot(h["epoch"], h["val_emotion_loss"], label="val")
        axes[0].set_title("Stage 1: Emotion Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    if stage2_hist:
        h = pd.DataFrame(stage2_hist)
        axes[1].plot(h["epoch"], h["train_suicide_mse"], label="train")
        axes[1].plot(h["epoch"], h["val_suicide_mse"], label="val")
        ylabel = "log1p(count) MSE" if log_transform_suicide else "Count MSE"
        axes[1].set_title(f"Stage 2: Suicide ({ylabel})")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    if joint_hist:
        h = pd.DataFrame(joint_hist)
        axes[2].plot(h["epoch"], h["train_total"], label="train total")
        axes[2].plot(h["epoch"], h["val_total_loss"], label="val total")
        axes[2].set_title("Joint Fine-tuning: Total Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] loss_curves.png")

    # ── 2. Predicted vs Actual (suicide count) ───────────────────
    joint_model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X, stress_mask, emo_seq, emo_mask, y_suicide in val_loader:
            X = X.to(device)
            stress_mask = stress_mask.to(device)
            emo_hist = emo_seq[:, :-1, :].to(device)
            emo_hist_mask = emo_mask[:, :-1].to(device)
            _, pred = joint_model(X, stress_mask, emo_hist, emo_hist_mask)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y_suicide.numpy())

    pred_arr = np.concatenate(all_pred, axis=0)   # (N, n_out)
    true_arr = np.concatenate(all_true, axis=0)   # (N, n_out)

    # Denormalize for display
    if log_transform_suicide:
        pred_disp = np.expm1(pred_arr)
        true_disp = np.expm1(true_arr)
    else:
        pred_disp, true_disp = pred_arr, true_arr

    n_out = len(target_cols)
    fig, axes = plt.subplots(1, n_out, figsize=(6 * n_out, 4))
    if n_out == 1:
        axes = [axes]

    for i, col in enumerate(target_cols):
        ax = axes[i]
        ax.scatter(true_disp[:, i], pred_disp[:, i], alpha=0.5, s=20)
        mn = min(true_disp[:, i].min(), pred_disp[:, i].min())
        mx = max(true_disp[:, i].max(), pred_disp[:, i].max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="y=x")
        rmse = np.sqrt(np.mean((pred_disp[:, i] - true_disp[:, i]) ** 2))
        ax.set_title(f"{_en(col)}  RMSE={rmse:.1f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pred_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] pred_vs_actual.png")

    # ── 3. Stage 1 emotion prediction vs GT ─────────────────────
    stage1_model = joint_model.stage1
    stage1_model.eval()
    all_pred_emo, all_true_emo = [], []
    with torch.no_grad():
        for X, stress_mask, emo_seq, emo_mask, _ in val_loader:
            X = X.to(device)
            stress_mask = stress_mask.to(device)
            pred_emo = stage1_model(X, stress_mask)
            all_pred_emo.append(pred_emo.cpu().numpy())
            all_true_emo.append(emo_seq[:, -1, :].numpy())  # GT emotion at t

    pred_emo = np.concatenate(all_pred_emo, axis=0)   # (N, 6)
    true_emo = np.concatenate(all_true_emo, axis=0)   # (N, 6)

    n_emo = len(emotion_cols)
    cols_per_row = 3
    n_rows = (n_emo + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(emotion_cols):
        ax = axes[i]
        ax.scatter(true_emo[:, i], pred_emo[:, i], alpha=0.5, s=15)
        mn = min(true_emo[:, i].min(), pred_emo[:, i].min())
        mx = max(true_emo[:, i].max(), pred_emo[:, i].max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        rmse = np.sqrt(np.mean((pred_emo[:, i] - true_emo[:, i]) ** 2))
        ax.set_title(f"{_en(col)}  RMSE={rmse:.4f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)

    for j in range(n_emo, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Stage 1: Emotion Prediction vs GT", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "emotion_pred_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] emotion_pred_vs_actual.png")


def run_pipeline(df: pd.DataFrame, feature_cols: Sequence[str], emotion_cols: Sequence[str], cfg: PipelineConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_root, exist_ok=True)

    # 타겟 컬럼 자동 탐지
    target_cols = None
    for candidates in [["남자", "여자"], ["자살자수"], ["자살사망자수"]]:
        if all(c in df.columns for c in candidates):
            target_cols = candidates
            break
    if target_cols is None:
        raise ValueError(f"타겟 컬럼을 찾을 수 없습니다. 컬럼 목록: {list(df.columns)}")
    n_out = len(target_cols)
    print(f"타겟 컬럼: {target_cols}  (n_out={n_out})")

    dataset, dl_train, dl_val = build_dataloaders(df, feature_cols, emotion_cols, cfg, target_cols)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_emotions = len(emotion_cols)

    stage1 = StressToEmotionModel(
        d_in=len(feature_cols),
        n_emotions=n_emotions,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        max_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        output_distribution=cfg.normalize_emotion_target,
    ).to(device)

    stage2 = EmotionToSuicideModel(
        n_emotions=n_emotions,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_out=n_out,
        emotion_lag=cfg.emotion_lag,
        dropout=cfg.dropout,
    ).to(device)

    print("\n=== Stage1 pretrain: stress -> emotion ===")
    stage1_hist = train_stage1(stage1, dl_train, dl_val, cfg, device)
    torch.save(stage1.state_dict(), os.path.join(cfg.save_root, "stage1_best.pt"))

    print("\n=== Stage2 pretrain: emotion -> suicide (GT emotion) ===")
    stage2_hist = train_stage2(stage2, dl_train, dl_val, cfg, device)
    torch.save(stage2.state_dict(), os.path.join(cfg.save_root, "stage2_best.pt"))

    joint_model = JointEmotionSuicideModel(stage1=stage1, stage2=stage2).to(device)
    print("\n=== Joint fine-tuning: predicted emotion -> suicide ===")
    joint_hist = train_joint(joint_model, dl_train, dl_val, cfg, device)
    torch.save(joint_model.state_dict(), os.path.join(cfg.save_root, "joint_best.pt"))

    save_histories(cfg.save_root, stage1_hist, stage2_hist, joint_hist)

    final_metrics = evaluate_joint(
        joint_model, dl_val, device, cfg.emotion_loss_type, cfg.lambda_emotion, cfg.lambda_suicide
    )
    print("\n[Final validation metrics]", final_metrics)

    meta = {
        "feature_cols": list(feature_cols),
        "emotion_cols": list(emotion_cols),
        "n_samples": len(dataset),
        "config": cfg.__dict__,
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }
    pd.Series(meta, dtype=object).to_json(os.path.join(cfg.save_root, "run_meta.json"), force_ascii=False, indent=2)

    print("\n[PLOT] 결과 시각화 저장 중 ...")
    plot_dir = os.path.join(cfg.save_root, "plots")
    plot_results(
        save_dir=plot_dir,
        stage1_hist=stage1_hist,
        stage2_hist=stage2_hist,
        joint_hist=joint_hist,
        joint_model=joint_model,
        val_loader=dl_val,
        target_cols=target_cols,
        emotion_cols=list(emotion_cols),
        device=device,
        log_transform_suicide=cfg.log_transform_suicide,
    )
    print(f"  저장 위치: {plot_dir}/")

    return {
        "stage1_model": stage1,
        "stage2_model": stage2,
        "joint_model": joint_model,
        "final_metrics": final_metrics,
    }


def infer_default_feature_cols(df: pd.DataFrame, emotion_cols: Sequence[str]) -> List[str]:
    exclude = {"날짜", "자살사망자수", "자살자수", "남자", "여자", *emotion_cols}
    return [c for c in df.columns if c not in exclude]


def main(
    emotion_cols: Optional[Sequence[str]] = None,
    cfg: Optional[PipelineConfig] = None,
):
    """실행 예시

    python -m emotion_suicide_trainer.supervised_pipeline

    실제 감정 분포 컬럼명이 아직 확정되지 않았으면 아래 emotion_cols를 수정하세요.
    """
    if cfg is None:
        cfg = PipelineConfig()
    if emotion_cols is None:
        emotion_cols = ["감정_분노", "감정_슬픔", "감정_불안", "감정_상처", "감정_당황", "감정_기쁨"]

    df = load_clean_csv(cfg.csv_path).sort_values("날짜").reset_index(drop=True)
    feature_cols = infer_default_feature_cols(df, emotion_cols)
    print(f"Using {len(feature_cols)} stress features")
    print(f"Emotion cols: {list(emotion_cols)}")
    return run_pipeline(df, feature_cols, emotion_cols, cfg)


if __name__ == "__main__":
    EMOTION_COLS = [
        "감정_분노",
        "감정_슬픔",
        "감정_불안",
        "감정_상처",
        "감정_당황",
        "감정_기쁨",
    ]
    main(emotion_cols=EMOTION_COLS)

