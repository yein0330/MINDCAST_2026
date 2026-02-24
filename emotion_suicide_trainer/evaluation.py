"""평가 및 시각화 모듈 — TwoStageSuicideModel 전용

주요 변경점 (vs suicide_trainer):
  - attention 쿼리 수: 2 (남/여) → 6 (분노/슬픔/불안/상처/당황/기쁨)
  - 모델 내부 접근 경로: model.stage1.* 사용
  - 신규 함수: visualize_emotion_vec, plot_stage2_weights
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .model import EMOTION_NAMES


# ============================================================
# Test: L=1~12 예측 (특정 target_date)
# ============================================================
@torch.no_grad()
def predict_L1_12_for_date(model, df, feature_cols, dataset, device, target_date, save_dir):
    target_dt = pd.to_datetime(target_date)
    input_end_dt = target_dt - pd.DateOffset(months=4)
    if input_end_dt.strftime("%Y-%m-%d") not in df["날짜"].values:
        raise ValueError(f"{input_end_dt.strftime('%Y-%m-%d')} 데이터 없음")

    end_idx = df.index[df["날짜"] == input_end_dt.strftime("%Y-%m-%d")][0]
    D = len(feature_cols)
    rows = []

    for L in range(1, 13):
        start_idx = end_idx - (L - 1)
        if start_idx < 0:
            continue
        X = (df.loc[start_idx:end_idx, feature_cols] - dataset.X_mu) / dataset.X_std
        pad_len = dataset.max_len - L
        pad = np.zeros((pad_len, D), np.float32)
        Xv = np.concatenate([pad, X.values.astype(np.float32)], axis=0)
        mask = np.array([True] * pad_len + [False] * L)

        Xv = torch.tensor(Xv, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)

        pred = model(Xv, mask).cpu().numpy().flatten()
        x_start, x_end = df.loc[start_idx, "날짜"], df.loc[end_idx, "날짜"]
        rows.append([L, x_start, x_end, target_date, pred[0], pred[1], D])

    os.makedirs(save_dir, exist_ok=True)
    out_df = pd.DataFrame(rows, columns=[
        "L", "x_start", "x_end", "target_date", "pred_male", "pred_female", "num_features"
    ])
    out_csv = os.path.join(save_dir, f"all_L_predictions_for_{target_date}.csv")
    out_df.to_csv(out_csv, index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(out_df["L"], out_df["pred_male"], marker='o', label="Male")
    plt.plot(out_df["L"], out_df["pred_female"], marker='s', label="Female")
    plt.xlabel("Sequence Length L (months)")
    plt.ylabel("Predicted suicides")
    plt.title(f"Predictions for {target_date}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(save_dir, f"pred_trend_{target_date}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved predictions -> {out_csv}")
    print(f"Saved plot -> {out_png}")
    return out_df


# ============================================================
# Attention Heatmap — 6 감정 쿼리
# ============================================================
@torch.no_grad()
def visualize_attention(model, df, feature_cols, dataset, device, save_dir, L=6):
    """6개 감정 쿼리의 Attention 가중치를 2×3 그리드로 시각화"""
    end_idx = len(df) - 5
    start_idx = end_idx - (L - 1)

    X = (df.loc[start_idx:end_idx, feature_cols] - dataset.X_mu) / dataset.X_std
    months = df.loc[start_idx:end_idx, "날짜"].tolist()
    pad_len = dataset.max_len - L
    Xv = np.concatenate([np.zeros((pad_len, len(feature_cols))),
                         X.values.astype(np.float32)], 0)
    mask = np.array([True] * pad_len + [False] * L)
    Xv = torch.tensor(Xv, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)

    _, attn = model(Xv, mask, return_attn=True)
    # attn: (1, 1, n_emotions, T) → (n_emotions, T)
    A = attn.squeeze(0).squeeze(0).cpu().numpy()

    n_emotions = len(EMOTION_NAMES)
    fig, axes = plt.subplots(2, 3, figsize=(14, 5))
    axes = axes.flatten()

    for i, (emo_name, ax) in enumerate(zip(EMOTION_NAMES, axes)):
        emo_map = A[i, -L:][None, :]  # (1, L)
        sns.heatmap(emo_map, ax=ax, cmap="Blues", cbar=True,
                    vmin=0, vmax=emo_map.max() + 1e-8)
        ax.set_title(f"{emo_name} Attention (L={L})", fontsize=10)
        ax.set_yticks([])
        ax.set_xticks(np.arange(L) + 0.5)
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("월")

    plt.suptitle(f"감정 쿼리별 Cross-Attention (L={L})", fontsize=13, y=1.02)
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"emotion_attn_heatmap_L{L}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved emotion attention heatmap -> {out}")


@torch.no_grad()
def visualize_attention_grid(model, df, feature_cols, dataset, device, save_dir,
                             L_list=range(1, 13)):
    """L=1~12 별 평균 감정 Attention을 4×3 그리드로 시각화"""
    cols, rows = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    axes = axes.flatten()

    for i, L in enumerate(L_list):
        end_idx = len(df) - 5
        start_idx = end_idx - (L - 1)
        X = (df.loc[start_idx:end_idx, feature_cols] - dataset.X_mu) / dataset.X_std
        pad_len = dataset.max_len - L
        Xv = np.concatenate([np.zeros((pad_len, len(feature_cols))),
                             X.values.astype(np.float32)], 0)
        mask = np.array([True] * pad_len + [False] * L)
        Xv = torch.tensor(Xv, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)

        _, attn = model(Xv, mask, return_attn=True)
        # 6개 감정 attention 평균 → (L,)
        A = attn.squeeze(0).squeeze(0).cpu().numpy()  # (n_emotions, T)
        mean_map = A[:, -L:].mean(axis=0)[None, :]    # (1, L)

        ax = axes[i]
        sns.heatmap(mean_map, ax=ax, cmap="Blues", cbar=False)
        ax.set_title(f"L={L}", fontsize=10)
        ax.set_yticks([])
        ax.set_xticks([])

    for j in range(i + 1, rows * cols):
        axes[j].axis("off")

    plt.suptitle("Mean Emotion Attention — L Grid (1~12)", y=1.02, fontsize=14)
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "emotion_attn_grid_L1_12.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved emotion attention grid -> {out}")


# ============================================================
# 신규: 감정 증폭 벡터 시각화 (Stage 1 출력)
# ============================================================
@torch.no_grad()
def visualize_emotion_vec(model, df, feature_cols, dataset, device, save_dir, L=6):
    """특정 L에서 Stage 1이 출력하는 감정 증폭 벡터를 막대 그래프로 시각화"""
    end_idx = len(df) - 5
    start_idx = end_idx - (L - 1)
    X = (df.loc[start_idx:end_idx, feature_cols] - dataset.X_mu) / dataset.X_std
    pad_len = dataset.max_len - L
    Xv = np.concatenate([np.zeros((pad_len, len(feature_cols))),
                         X.values.astype(np.float32)], 0)
    mask = np.array([True] * pad_len + [False] * L)
    Xv = torch.tensor(Xv, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)

    _, emotion_vec = model(Xv, mask, return_emotion=True)
    emo = emotion_vec.squeeze(0).cpu().numpy()  # (n_emotions,)

    plt.figure(figsize=(7, 4))
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in emo]
    plt.bar(EMOTION_NAMES, emo, color=colors, edgecolor="none")
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.title(f"감정 증폭 벡터 (Stage 1 출력, L={L})", fontsize=12)
    plt.ylabel("activation")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"emotion_vec_L{L}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved emotion vec -> {out}")


# ============================================================
# 신규: Stage 2 가중치 해석
# ============================================================
def plot_stage2_weights(model, save_dir):
    """Stage 2 첫 번째 선형층 가중치를 통해
    각 감정이 남/여 자살자 수에 미치는 직접 영향력을 시각화."""
    os.makedirs(save_dir, exist_ok=True)

    male_w = model.stage2.male_head[0].weight.detach().cpu().numpy()    # (d_hidden, n_emotions)
    female_w = model.stage2.female_head[0].weight.detach().cpu().numpy()  # (d_hidden, n_emotions)

    # 각 감정의 평균 절댓값 영향력
    male_imp = np.abs(male_w).mean(axis=0)    # (n_emotions,)
    female_imp = np.abs(female_w).mean(axis=0)

    df_imp = pd.DataFrame({
        "emotion": EMOTION_NAMES,
        "Male": male_imp,
        "Female": female_imp,
    })

    plt.figure(figsize=(8, 5))
    x = np.arange(len(EMOTION_NAMES))
    w = 0.35
    plt.bar(x - w / 2, df_imp["Male"], width=w, label="Male", color="#1f77b4", edgecolor="none")
    plt.bar(x + w / 2, df_imp["Female"], width=w, label="Female", color="#ff7f0e", edgecolor="none")
    plt.xticks(x, EMOTION_NAMES)
    plt.ylabel("mean |weight|")
    plt.title("Stage 2: 감정별 자살자 수 영향력 (가중치 절댓값 평균)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, "stage2_emotion_weights.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    df_imp.to_csv(os.path.join(save_dir, "stage2_emotion_weights.csv"), index=False)
    print(f"Saved Stage 2 weights -> {out}")


# ============================================================
# Feature Importance — |dy/dx| (autograd.grad)
# ============================================================
def _get_kv(model, Xv):
    """Stage 1 Key/Value 표현 추출 (gradient 계산용)"""
    return model.stage1.norm_kv(model.stage1.posenc(model.stage1.proj(Xv)))


def get_timestep_feature_gradient(model, df, feature_cols, dataset,
                                  device, L=6, t_index=0):
    end_idx = len(df) - 5
    start_idx = end_idx - (L - 1)

    X = (df.loc[start_idx:end_idx, feature_cols] - dataset.X_mu) / dataset.X_std
    D = len(feature_cols)
    pad_len = dataset.max_len - L
    pad = np.zeros((pad_len, D), np.float32)
    Xv_np = np.concatenate([pad, X.values.astype(np.float32)], 0)
    Xv = torch.tensor(Xv_np, dtype=torch.float32).unsqueeze(0).to(device)
    Xv.requires_grad_(True)

    mask = torch.tensor(
        [True] * pad_len + [False] * L, dtype=torch.bool
    ).unsqueeze(0).to(device)

    with torch.enable_grad():
        kv = _get_kv(model, Xv)
        obj = kv[0, pad_len + t_index].norm()
        obj.backward()

    grad = Xv.grad[0, pad_len + t_index].detach().cpu().numpy()
    return np.abs(grad)


def compute_feature_importance_for_timestep(model, df, feature_cols, dataset,
                                            device, L=6, time_index=0):
    end_idx = len(df) - 5
    start_idx = end_idx - (L - 1)

    X = (df.loc[start_idx:end_idx, feature_cols] - dataset.X_mu) / dataset.X_std
    pad_len = dataset.max_len - L
    Xv_np = np.concatenate([
        np.zeros((pad_len, len(feature_cols))),
        X.values.astype(np.float32)
    ], 0)

    Xv = torch.tensor(Xv_np, dtype=torch.float32).unsqueeze(0).to(device)
    Xv.requires_grad_(True)
    mask = torch.tensor(
        [True] * pad_len + [False] * L, dtype=torch.bool
    ).unsqueeze(0).to(device)

    model.eval()
    with torch.enable_grad():
        kv = _get_kv(model, Xv)
        kv_index = pad_len + time_index
        obj = kv[0, kv_index].norm()
        obj.backward()

    if Xv.grad is None:
        raise RuntimeError("Xv.grad is None — grad graph disconnected")

    grad_t = Xv.grad.squeeze(0)[kv_index].detach().cpu().numpy()
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": np.abs(grad_t)
    }).sort_values("importance", ascending=False)
    return imp


def compute_feature_importance_for_L(model, df, feature_cols, dataset, device, L):
    """특정 시퀀스 길이(L)에서 |dy/dx| 계산"""
    model.eval()
    end_idx = len(df) - 5
    start_idx = end_idx - (L - 1)
    X_df = df[feature_cols].iloc[start_idx:end_idx + 1]
    X = ((X_df - dataset.X_mu) / dataset.X_std).values.astype(np.float32)

    pad_len = dataset.max_len - L
    pad = np.zeros((pad_len, len(feature_cols)), np.float32)
    Xv = np.concatenate([pad, X], axis=0)

    X_tensor = torch.tensor(Xv, dtype=torch.float32).unsqueeze(0).to(device)
    X_tensor.requires_grad_(True)
    mask = torch.tensor(
        [True] * pad_len + [False] * L, dtype=torch.bool
    ).unsqueeze(0).to(device)

    y_hat = model(X_tensor, mask)
    if not y_hat.requires_grad:
        raise RuntimeError("y_hat does not require grad — gradient graph disconnected.")

    grads = torch.autograd.grad(
        outputs=y_hat.sum(), inputs=X_tensor,
        retain_graph=False, create_graph=False
    )[0]

    g = grads.abs().mean(dim=1).squeeze(0).detach().cpu().numpy()
    imp = pd.DataFrame({"feature": feature_cols, "importance": g})
    return imp.sort_values("importance", ascending=True)


def plot_feature_importance(imp, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(imp))
    bars = plt.barh(imp["feature"], imp["importance"], color=colors, edgecolor="none", height=0.55)
    for b in bars:
        w = b.get_width()
        plt.text(w + imp["importance"].max() * 0.02, b.get_y() + b.get_height() / 2,
                 f"{w:.4f}", va="center", fontsize=9, color="#333")
    plt.title("Feature Importance (|dy/dx| mean)", fontsize=13, pad=12)
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.grid(axis="x", linestyle="--", alpha=0.45)
    plt.tight_layout()
    out = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance -> {out}")


# ============================================================
# Feature Sensitivity: ±5% 변화 → Δy & Elasticity
# ============================================================
@torch.no_grad()
def simulate_feature_sensitivity(model, df, feature_cols, dataset, device,
                                 pct_list=(+0.05, -0.05)):
    end_idx = len(df) - 5
    start_idx = end_idx - 11
    D = len(feature_cols)

    base_X = (df.loc[start_idx:end_idx, feature_cols] - dataset.X_mu) / dataset.X_std
    pad_len = dataset.max_len - 12
    pad = np.zeros((pad_len, D), np.float32)
    base_input = np.concatenate([pad, base_X.values.astype(np.float32)], 0)
    mask = np.array([True] * pad_len + [False] * 12)

    base_in = torch.tensor(base_input, dtype=torch.float32).unsqueeze(0).to(device)
    mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
    base_pred = model(base_in, mask_t).cpu().numpy().flatten()
    base_male, base_female = base_pred.tolist()

    rows = []
    for idx, f in enumerate(feature_cols):
        for p in pct_list:
            X_mod = base_X.copy()
            X_mod.iloc[:, idx] *= (1.0 + p)
            Xv = np.concatenate([pad, X_mod.values.astype(np.float32)], 0)
            Xv = torch.tensor(Xv, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(Xv, mask_t).cpu().numpy().flatten()
            d = pred - base_pred
            rows.append({
                "feature": f,
                "change": f"{int(p * 100)}%",
                "Δmale": float(d[0]),
                "Δfemale": float(d[1]),
                "male_elasticity": float((d[0] / max(base_male, 1e-6)) / p),
                "female_elasticity": float((d[1] / max(base_female, 1e-6)) / p),
                "base_male": base_male,
                "base_female": base_female,
            })
    return pd.DataFrame(rows)


def plot_sensitivity(sens_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for col, title, fname in [
        ("Δmale", "ΔMale (±5% feature change)", "sensitivity_male.png"),
        ("Δfemale", "ΔFemale (±5% feature change)", "sensitivity_female.png"),
    ]:
        plt.figure(figsize=(10, 6))
        order = sens_df.groupby("feature")[col].mean().sort_values().index
        sns.barplot(data=sens_df, x=col, y="feature", hue="change",
                    order=order, palette="coolwarm", edgecolor="none")
        plt.title(title, fontsize=13, pad=12)
        plt.xlabel("Δ predicted suicides")
        plt.ylabel("feature")
        plt.grid(axis="x", linestyle="--", alpha=0.45)
        plt.tight_layout()
        out = os.path.join(save_dir, fname)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved sensitivity plot -> {out}")


# ============================================================
# Extended Visualization Suite
# ============================================================
def plot_train_val_curve(train_losses, val_losses, save_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve (Loss per Epoch)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, "train_val_curve.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved train/val curve -> {out}")


def plot_feature_importance_dual(model, df, feature_cols, dataset, device, save_dir):
    """남/여 각각에 대한 Feature Importance를 별도 계산하여 비교"""
    model.eval()
    X_df = df[feature_cols].iloc[-dataset.max_len:]
    X = torch.tensor(
        ((X_df - dataset.X_mu) / dataset.X_std).values,
        dtype=torch.float32, requires_grad=True
    ).unsqueeze(0).to(device)
    mask = torch.zeros((1, X.shape[1]), dtype=torch.bool).to(device)

    y_hat = model(X, mask)

    # 남자 gradient
    grads_male = torch.autograd.grad(
        outputs=y_hat[:, 0].sum(), inputs=X, retain_graph=True
    )[0]
    # 여자 gradient
    grads_female = torch.autograd.grad(
        outputs=y_hat[:, 1].sum(), inputs=X, retain_graph=False
    )[0]

    male_imp = grads_male[0, :, :].abs().mean(0).detach().cpu().numpy()
    female_imp = grads_female[0, :, :].abs().mean(0).detach().cpu().numpy()

    df_imp = pd.DataFrame({"feature": feature_cols, "Male": male_imp, "Female": female_imp})
    df_imp = df_imp.sort_values("Male", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp.melt(id_vars="feature", value_vars=["Male", "Female"]),
                x="value", y="feature", hue="variable", palette=["#1f77b4", "#ff7f0e"])
    plt.title("Feature Importance by Gender (|dy/dx| mean)")
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.tight_layout()
    out = os.path.join(save_dir, "feature_importance_gender_split.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved gender-split importance -> {out}")


@torch.no_grad()
def plot_time_feature_attention(model, df, feature_cols, dataset, device, save_dir, L=6):
    """6개 감정 쿼리의 시간별 Attention을 6행 히트맵으로 시각화"""
    end_idx = len(df) - 5
    start_idx = end_idx - (L - 1)
    X = (df.loc[start_idx:end_idx, feature_cols] - dataset.X_mu) / dataset.X_std
    months = df.loc[start_idx:end_idx, "날짜"].tolist()

    Xv = torch.tensor(X.values, dtype=torch.float32).unsqueeze(0).to(device)
    pad = torch.zeros((1, dataset.max_len - L, len(feature_cols)), device=device)
    Xv = torch.cat([pad, Xv], dim=1)
    mask = torch.tensor(
        [True] * (dataset.max_len - L) + [False] * L,
        dtype=torch.bool, device=device
    ).unsqueeze(0)

    _, attn = model(Xv, mask, return_attn=True)
    A = attn.squeeze(0).squeeze(0).cpu().numpy()  # (n_emotions, T)

    n_emotions = len(EMOTION_NAMES)
    fig, axes = plt.subplots(n_emotions, 1, figsize=(10, n_emotions * 1.2))
    cmaps = ["Reds", "Blues", "Greens", "Purples", "Oranges", "YlOrBr"]

    for i, (emo_name, ax, cmap) in enumerate(zip(EMOTION_NAMES, axes, cmaps)):
        emo_map = A[i, -L:].reshape(1, L)
        sns.heatmap(emo_map, cmap=cmap, ax=ax, cbar=False)
        ax.set_ylabel(emo_name, rotation=0, labelpad=35, va="center", fontsize=9)
        ax.set_yticks([])
        ax.set_xticks([])

    axes[-1].set_xticks(np.arange(L) + 0.5)
    axes[-1].set_xticklabels(months, rotation=45, ha="right", fontsize=8)

    plt.suptitle(f"감정 쿼리 Temporal Attention (L={L})", fontsize=12)
    plt.tight_layout()
    out = os.path.join(save_dir, f"emotion_time_attn_L{L}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved emotion time-feature attention -> {out}")


def plot_sensitivity_elasticity(sens_df, save_dir):
    sens_df = sens_df.copy()
    sens_df["male_elasticity_abs"] = sens_df["male_elasticity"].abs()
    sens_df["female_elasticity_abs"] = sens_df["female_elasticity"].abs()
    sens_df = sens_df.groupby("feature")[
        ["male_elasticity_abs", "female_elasticity_abs"]
    ].mean().reset_index()
    sens_df = sens_df.sort_values("male_elasticity_abs", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=sens_df.melt(id_vars="feature"),
                x="value", y="feature", hue="variable", palette=["#4c72b0", "#dd8452"])
    plt.title("Feature Elasticity (Sensitivity of Δy to ±5%)")
    plt.xlabel("mean elasticity (abs)")
    plt.ylabel("feature")
    plt.tight_layout()
    out = os.path.join(save_dir, "feature_elasticity.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved elasticity plot -> {out}")


def plot_correlation_heatmap(df, feature_cols, save_dir):
    plt.figure(figsize=(10, 8))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation between socio-economic indicators")
    plt.tight_layout()
    out = os.path.join(save_dir, "correlation_heatmap.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap -> {out}")


@torch.no_grad()
def plot_3d_attention_feature_multi(model, df, feature_cols, dataset, device, save_dir, L=6):
    """6개 감정 쿼리의 Attention을 3D 라인 플롯으로 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    T = len(df)
    end_idx_list = list(range(T - 9, T - 4))

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    for end_idx in end_idx_list:
        start_idx = end_idx - (L - 1)
        if start_idx < 0:
            continue

        X = (df.loc[start_idx:end_idx, feature_cols] - dataset.X_mu) / dataset.X_std
        pad_len = dataset.max_len - L
        Xv = np.concatenate([
            np.zeros((pad_len, len(feature_cols))),
            X.values.astype(np.float32)
        ], 0)
        Xv = torch.tensor(Xv, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.tensor(
            [True] * pad_len + [False] * L, dtype=torch.bool
        ).unsqueeze(0).to(device)

        _, attn = model(Xv, mask, return_attn=True)
        A = attn.squeeze(0).squeeze(0).cpu().numpy()  # (n_emotions, T)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection="3d")
        xs = np.arange(L)

        for i, (emo_name, col) in enumerate(zip(EMOTION_NAMES, colors)):
            ys = np.full(L, i, dtype=float)
            ax.plot(xs, ys, A[i, -L:], marker='o', color=col, label=emo_name)

        ax.set_xlabel("Time index")
        ax.set_ylabel("Emotion")
        ax.set_zlabel("Attention weight")
        ax.set_yticks(range(len(EMOTION_NAMES)))
        ax.set_yticklabels(EMOTION_NAMES, fontsize=7)
        x_start = df.loc[start_idx, "날짜"]
        x_end = df.loc[end_idx, "날짜"]
        ax.set_title(f"Emotion Attention Surface\nWindow: {x_start} ~ {x_end}")
        ax.legend(loc="upper left", fontsize=7)
        plt.tight_layout()
        out = os.path.join(save_dir, f"emotion_attn_lines_L{L}_end_{x_end}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out}")


def plot_attention_feature_contribution_multi(model, df, feature_cols, dataset,
                                              device, save_dir, L=6):
    """감정별 × 타임스텝별 Feature Contribution (Attention gradient)

    각 감정 쿼리의 Attention 가중치에 대한 입력 피처의 gradient를 시각화.
    '어떤 스트레스 지표가 어느 감정을 얼마나 활성화하는가'를 해석 가능.
    """
    os.makedirs(save_dir, exist_ok=True)
    T = len(df)
    end_idx_list = list(range(T - 9, T - 4))

    for end_idx in end_idx_list:
        start_idx = end_idx - (L - 1)
        if start_idx < 0:
            continue

        X_win_df = (df.loc[start_idx:end_idx, feature_cols] - dataset.X_mu) / dataset.X_std
        X_win_np = X_win_df.values.astype(np.float32)
        D = X_win_np.shape[1]
        pad_len = dataset.max_len - L

        months = df.loc[start_idx:end_idx, "날짜"].tolist()
        x_start, x_end = months[0], months[-1]

        X_core = torch.tensor(X_win_np, dtype=torch.float32, device=device, requires_grad=True)
        pad = torch.zeros((pad_len, D), dtype=torch.float32, device=device)
        Xv = torch.cat([pad, X_core], dim=0).unsqueeze(0)
        mask = torch.tensor(
            [True] * pad_len + [False] * L, dtype=torch.bool, device=device
        ).unsqueeze(0)

        model.eval()
        with torch.enable_grad():
            y_hat, attn = model(Xv, mask, return_attn=True)
            A = attn.squeeze(0).squeeze(0)  # (n_emotions, T)

            for emo_idx, emo_name in enumerate(EMOTION_NAMES):
                for t_step in range(L):
                    kv_index = pad_len + t_step
                    obj = A[emo_idx, kv_index]

                    model.zero_grad(set_to_none=True)
                    if X_core.grad is not None:
                        X_core.grad.zero_()

                    obj.backward(retain_graph=True)

                    grad_core = X_core.grad.detach().cpu().numpy()
                    grad_t = grad_core[t_step]

                    imp = pd.DataFrame({
                        "feature": feature_cols,
                        "importance": np.abs(grad_t),
                        "signed_grad": grad_t,
                    }).sort_values("importance", ascending=True)

                    csv_name = f"feature_contrib_L{L}_end_{x_end}_t{t_step}_{emo_name}.csv"
                    imp.to_csv(os.path.join(save_dir, csv_name), index=False)

                    plt.figure(figsize=(9, 6))
                    colors = sns.color_palette("viridis", len(imp))
                    bars = plt.barh(imp["feature"], imp["importance"],
                                    color=colors, edgecolor="none", height=0.55)
                    for b, sg in zip(bars, imp["signed_grad"]):
                        w = b.get_width()
                        sign = "+" if sg >= 0 else "-"
                        plt.text(
                            w + imp["importance"].max() * 0.02,
                            b.get_y() + b.get_height() / 2,
                            f"{sign}{abs(sg):.4f}", va="center", fontsize=8, color="#333",
                        )
                    plt.title(
                        f"[L={L}] Window {x_start} ~ {x_end}\n"
                        f"{emo_name} attention @ t={t_step}",
                        fontsize=12,
                    )
                    plt.xlabel("importance = |d(attention) / d(feature)|")
                    plt.ylabel("feature")
                    plt.grid(axis="x", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    png_name = f"feature_contrib_L{L}_end_{x_end}_t{t_step}_{emo_name}.png"
                    out_path = os.path.join(save_dir, png_name)
                    plt.savefig(out_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"contrib plot -> {out_path}")
