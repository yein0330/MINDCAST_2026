"""트레이너 모듈 — 학습/평가 루프 및 메인 파이프라인

2단계 모델 (TwoStageSuicideModel) 학습:
  - base_data.csv에 감정 컬럼(감정_분노~기쁨)이 있으면:
      Stage 1 MSE (감정 예측) + Stage 2 MSE (자살자 수) → weighted sum
  - 감정 컬럼 없으면:
      Stage 2 MSE (자살자 수) 단독 → end-to-end

하이퍼파라미터:
  EMOTION_LOSS_WEIGHT  : Stage 1 감정 loss 가중치 α
                         (0이면 감정 supervision 미사용)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .utils import (
    CSV_PATH, SAVE_ROOT, TARGET_DATE, MAX_LEN, BATCH_SIZE, EPOCHS, LR, SEED,
    set_seed, set_korean_font, load_clean_csv,
)
from .dataset import FullLCombinationDataset, EMOTION_COLS_DEFAULT
from .model import TwoStageSuicideModel, EMOTION_NAMES
from .evaluation import (
    predict_L1_12_for_date,
    visualize_attention,
    visualize_attention_grid,
    compute_feature_importance_for_L,
    compute_feature_importance_for_timestep,
    plot_attention_feature_contribution_multi,
    simulate_feature_sensitivity,
    plot_sensitivity,
    plot_feature_importance_dual,
    plot_time_feature_attention,
    plot_sensitivity_elasticity,
    plot_correlation_heatmap,
    plot_3d_attention_feature_multi,
    visualize_emotion_vec,
    plot_stage2_weights,
    plot_train_val_curve,
)

# Stage 1 감정 loss 가중치 (0 → 감정 supervision 없음)
EMOTION_LOSS_WEIGHT = 0.3


# ============================================================
# Train / Eval 루프
# ============================================================
def train_epoch(model, loader, optim, device, emotion_loss_weight=EMOTION_LOSS_WEIGHT):
    """
    배치에 y_emotion이 포함된 경우 (4-tuple):
        loss = (1 - α) * MSE(y_hat, y_suicide) + α * MSE(emotion_vec, y_emotion)
    포함되지 않은 경우 (3-tuple):
        loss = MSE(y_hat, y_suicide)
    """
    model.train()
    mse = nn.MSELoss()
    tot = 0.0
    for batch in loader:
        has_emotion = len(batch) == 4
        if has_emotion:
            X, mask, y_suicide, y_emotion = batch
            X, mask = X.to(device), mask.to(device)
            y_suicide, y_emotion = y_suicide.to(device), y_emotion.to(device)
        else:
            X, mask, y_suicide = batch
            X, mask, y_suicide = X.to(device), mask.to(device), y_suicide.to(device)

        optim.zero_grad(set_to_none=True)

        if has_emotion and emotion_loss_weight > 0:
            y_hat, emotion_vec = model(X, mask, return_emotion=True)
            loss_suicide = mse(y_hat, y_suicide)
            loss_emotion = mse(emotion_vec, y_emotion)
            loss = (1 - emotion_loss_weight) * loss_suicide + emotion_loss_weight * loss_emotion
        else:
            y_hat = model(X, mask)
            loss = mse(y_hat, y_suicide)

        loss.backward()
        optim.step()
        tot += loss.item() * X.size(0)

    return tot / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    """검증: 자살자 수 MSE만으로 평가 (해석 가능한 지표)"""
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    tot = 0.0
    for batch in loader:
        X, mask, y_suicide = batch[0], batch[1], batch[2]
        X, mask, y_suicide = X.to(device), mask.to(device), y_suicide.to(device)
        tot += mse(model(X, mask), y_suicide).item()
    return tot / len(loader.dataset)


# ============================================================
# Main Pipeline
# ============================================================
def main():
    set_seed(SEED)
    set_korean_font()

    # ---- Load CSV ----
    df = load_clean_csv(CSV_PATH).sort_values("날짜").reset_index(drop=True)

    # ---- 타겟 컬럼 자동 탐지 ----
    # 우선순위: [남자, 여자] > [자살자수] > [자살사망자수]
    for candidates in [["남자", "여자"], ["자살자수"], ["자살사망자수"]]:
        if all(c in df.columns for c in candidates):
            target_cols = candidates
            break
    else:
        raise ValueError(
            f"타겟 컬럼을 찾을 수 없습니다. 컬럼 목록: {list(df.columns)}"
        )
    print(f"타겟 컬럼: {target_cols}  (n_out={len(target_cols)})")

    # ---- 컬럼 분류 ----
    exclude = {"날짜"} | set(target_cols)

    # 감정 컬럼 (data_loader.py가 생성한 감정_* 컬럼)
    emotion_cols = [c for c in df.columns if c in EMOTION_COLS_DEFAULT]

    # 사회경제 지표 컬럼 (감정 컬럼 제외)
    feature_cols = [c for c in df.columns if c not in exclude and c not in emotion_cols]

    print(f"Feature cols ({len(feature_cols)}): {feature_cols}")
    print(f"Emotion cols ({len(emotion_cols)}): {emotion_cols}")
    use_emotion_supervision = len(emotion_cols) == len(EMOTION_NAMES)
    print(f"Stage 1 감정 supervision: {'ON' if use_emotion_supervision else 'OFF'}")

    # ---- Dataset / Split ----
    dataset = FullLCombinationDataset(
        df,
        feature_cols=feature_cols,
        emotion_cols=emotion_cols if use_emotion_supervision else None,
        target_cols=target_cols,
        max_seq_len=MAX_LEN,
        pred_offset=4,
    )
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    train_ds = Subset(dataset, range(0, n_train))
    val_ds = Subset(dataset, range(n_train, n_total))
    dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwoStageSuicideModel(
        d_in=len(feature_cols),
        d_model=256,
        n_heads=8,
        n_emotions=6,
        max_len=MAX_LEN,
        dropout=0.1,
        d_hidden=128,
        n_out=len(target_cols),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---- Train ----
    train_losses, val_losses = [], []
    alpha = EMOTION_LOSS_WEIGHT if use_emotion_supervision else 0.0

    for ep in range(1, EPOCHS + 1):
        tr = train_epoch(model, dl_train, optim, device, emotion_loss_weight=alpha)
        va = eval_epoch(model, dl_val, device)
        train_losses.append(tr)
        val_losses.append(va)
        print(f"[Epoch {ep:03d}] Train={tr:.2f} | Val={va:.2f}")

    # ---- Save ----
    os.makedirs(SAVE_ROOT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_ROOT, "model_final.pt"))

    # ---- Analysis ----
    test_dir = os.path.join(SAVE_ROOT, f"test_{TARGET_DATE.replace('-', '')}")
    predict_L1_12_for_date(model, df, feature_cols, dataset, device, TARGET_DATE, test_dir)

    an_dir = os.path.join(SAVE_ROOT, "analysis")
    os.makedirs(an_dir, exist_ok=True)

    plot_train_val_curve(train_losses, val_losses, an_dir)

    visualize_attention(model, df, feature_cols, dataset, device, an_dir, L=6)
    visualize_attention_grid(model, df, feature_cols, dataset, device, an_dir, L_list=range(1, 13))
    visualize_emotion_vec(model, df, feature_cols, dataset, device, an_dir, L=6)
    plot_stage2_weights(model, an_dir)

    L_dir = os.path.join(an_dir, "feature_importance_by_L")
    os.makedirs(L_dir, exist_ok=True)
    for L in range(1, 13):
        imp_L = compute_feature_importance_for_L(model, df, feature_cols, dataset, device, L)
        out_csv_L = os.path.join(L_dir, f"feature_importance_L{L:02d}.csv")
        imp_L.to_csv(out_csv_L, index=False)

        plt.figure(figsize=(9, 5))
        colors = sns.color_palette("viridis", len(imp_L))
        plt.barh(imp_L["feature"], imp_L["importance"], color=colors, height=0.55)
        plt.title(f"Feature Importance (|dy/dx| mean) — L={L}", fontsize=12)
        plt.xlabel("importance")
        plt.ylabel("feature")
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(L_dir, f"feature_importance_L{L:02d}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    compute_feature_importance_for_timestep(
        model, df, feature_cols, dataset, device, L=6, time_index=0
    )
    plot_attention_feature_contribution_multi(
        model, df, feature_cols, dataset, device,
        save_dir=os.path.join(an_dir, "feature_contrib_multi"),
        L=6,
    )

    sens_df = simulate_feature_sensitivity(model, df, feature_cols, dataset, device,
                                           pct_list=(+0.05, -0.05))
    sens_df.to_csv(os.path.join(an_dir, "feature_sensitivity.csv"), index=False)
    plot_sensitivity(sens_df, an_dir)
    plot_feature_importance_dual(model, df, feature_cols, dataset, device, an_dir)
    plot_time_feature_attention(model, df, feature_cols, dataset, device, an_dir, L=6)
    plot_sensitivity_elasticity(sens_df, an_dir)
    plot_correlation_heatmap(df, feature_cols, an_dir)
    plot_3d_attention_feature_multi(model, df, feature_cols, dataset, device, an_dir, L=6)

    print(f"\nAll done. Outputs saved under: {SAVE_ROOT}/")


if __name__ == "__main__":
    main()
