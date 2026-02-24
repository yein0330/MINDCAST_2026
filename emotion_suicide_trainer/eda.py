"""EDA — base_data.csv 탐색적 데이터 분석

생성 플롯:
  1. suicide_trend.png        — 자살자수 월별 추세
  2. emotion_counts.png       — 감정별 댓글수 추세 (비율 × 댓글수)
  3. emotion_ratios.png       — 감정별 비율 추세
  4. suicide_vs_emotion.png   — 자살자수 vs 각 감정 raw count (scatter + time)
  5. socioeco_vs_emotion.png  — 주요 사회경제 지표 vs 감정 raw count (line overlay)
  6. corr_suicide_emotion.png — 자살자수 ↔ 감정 상관 (lag 0~6개월)
  7. corr_heatmap.png         — 사회경제 지표 ↔ 감정 상관 히트맵

Usage:
    cd /home/yein38/mindcastlib_trainer
    python -m emotion_suicide_trainer.eda
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# ── 설정 ───────────────────────────────────────────────────────────────────
CSV_PATH = "/home/yein38/mindcastlib_trainer/data/base/base_data.csv"
SAVE_DIR = "/home/yein38/mindcastlib_trainer/results/eda"

EMOTION_COLS = ["감정_분노", "감정_슬픔", "감정_불안", "감정_상처", "감정_당황", "감정_기쁨"]
EMOTION_EN   = ["Anger", "Sadness", "Anxiety", "Hurt", "Embarrassment", "Joy"]
EMOTION_COLORS = ["#e74c3c", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#f1c40f"]

# 시각화할 사회경제 지표 (한글 → 영어)
SOCIOECO = {
    "실업률(%)":          "Unemployment Rate (%)",
    "소비자물가상승률(%)": "CPI Inflation (%)",
    "고용률(%)":          "Employment Rate (%)",
    "임금총액":           "Total Wages",
    "가계대출":           "Household Loans",
    "근로시간":           "Working Hours",
    "환자수(총계)":       "Total Patients",
    "경제활동참가율(%)":  "Labor Participation (%)",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         120,
})


# ── 데이터 로드 ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.sort_values("날짜").reset_index(drop=True)

    # 감정 raw count = 비율 × 댓글수
    for col, en in zip(EMOTION_COLS, EMOTION_EN):
        df[f"cnt_{en}"] = df[col] * df["댓글수"]

    return df


def _date_fmt(ax, df):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)


# ── 1. 자살자수 월별 추세 ───────────────────────────────────────────────────
def plot_suicide_trend(df, save_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["날짜"], df["자살사망자수"], marker="o", ms=4, color="#c0392b", lw=2)
    ax.fill_between(df["날짜"], df["자살사망자수"], alpha=0.15, color="#c0392b")
    ax.set_title("Monthly Suicide Deaths", fontsize=14)
    ax.set_ylabel("Count")
    _date_fmt(ax, df)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "suicide_trend.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] suicide_trend.png")


# ── 2. 감정 raw count 추세 ──────────────────────────────────────────────────
def plot_emotion_counts(df, save_dir):
    count_cols = [f"cnt_{en}" for en in EMOTION_EN]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i, (col, en, color) in enumerate(zip(count_cols, EMOTION_EN, EMOTION_COLORS)):
        ax = axes[i]
        ax.plot(df["날짜"], df[col], color=color, lw=2, marker="o", ms=3)
        ax.fill_between(df["날짜"], df[col], alpha=0.15, color=color)
        ax.set_title(f"{en} (comment count)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        _date_fmt(ax, df)

    fig.suptitle("Emotion Raw Counts Over Time (ratio × total comments)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "emotion_counts.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] emotion_counts.png")


# ── 3. 감정 비율 추세 ───────────────────────────────────────────────────────
def plot_emotion_ratios(df, save_dir):
    fig, ax = plt.subplots(figsize=(13, 5))
    for col, en, color in zip(EMOTION_COLS, EMOTION_EN, EMOTION_COLORS):
        ax.plot(df["날짜"], df[col], label=en, color=color, lw=1.8, marker="o", ms=3)
    ax.set_title("Emotion Ratios Over Time", fontsize=13)
    ax.set_ylabel("Ratio (sum=1)")
    ax.legend(ncol=3, fontsize=9)
    _date_fmt(ax, df)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "emotion_ratios.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] emotion_ratios.png")


# ── 4. 자살자수 vs 감정 raw count ───────────────────────────────────────────
def plot_suicide_vs_emotion(df, save_dir):
    count_cols = [f"cnt_{en}" for en in EMOTION_EN]
    n = len(EMOTION_EN)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (col, en, color) in enumerate(zip(count_cols, EMOTION_EN, EMOTION_COLORS)):
        ax = axes[i]
        sc = ax.scatter(
            df[col], df["자살사망자수"],
            c=df.index, cmap="viridis", s=40, alpha=0.8, edgecolors="none"
        )
        # 추세선
        m, b = np.polyfit(df[col].fillna(0), df["자살사망자수"], 1)
        xs = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(xs, m * xs + b, "r--", lw=1.5, alpha=0.7)

        r = df[[col, "자살사망자수"]].dropna().corr().iloc[0, 1]
        ax.set_title(f"{en}  (r={r:.3f})", fontsize=11)
        ax.set_xlabel(f"{en} comment count")
        ax.set_ylabel("Suicide Deaths")
        ax.grid(alpha=0.3)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Month index (earlier=dark, later=bright)")
    fig.suptitle("Suicide Deaths vs Emotion Raw Counts", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "suicide_vs_emotion.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] suicide_vs_emotion.png")


# ── 5. 사회경제 지표 vs 감정 raw count (dual-axis time series) ─────────────
def plot_socioeco_vs_emotion(df, save_dir):
    count_cols = [f"cnt_{en}" for en in EMOTION_EN]
    n_socio = len(SOCIOECO)
    n_emo   = len(EMOTION_EN)

    fig, axes = plt.subplots(n_socio, 1, figsize=(13, 4 * n_socio), sharex=True)

    for ax_main, (kr_col, en_label) in zip(axes, SOCIOECO.items()):
        if kr_col not in df.columns:
            ax_main.set_visible(False)
            continue

        # z-score 정규화
        def znorm(s):
            return (s - s.mean()) / (s.std() + 1e-8)

        # 감정 중에서 자살자수와 가장 상관 높은 2개 선택
        corrs = {en: abs(df[f"cnt_{en}"].corr(df["자살사망자수"])) for en in EMOTION_EN}
        top2  = sorted(corrs, key=corrs.get, reverse=True)[:2]

        ax_main.plot(df["날짜"], znorm(df[kr_col]), color="#2c3e50", lw=2,
                     label=en_label, zorder=3)
        ax_main.set_ylabel(f"{en_label}\n(z-score)", fontsize=9)

        ax2 = ax_main.twinx()
        for en, color in zip(top2, ["#e74c3c", "#3498db"]):
            ax2.plot(df["날짜"], znorm(df[f"cnt_{en}"]), color=color, lw=1.5,
                     linestyle="--", alpha=0.8, label=f"{en} cnt (z)")
        ax2.set_ylabel("Emotion (z-score)", fontsize=9)

        lines1, labels1 = ax_main.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_main.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        ax_main.grid(alpha=0.2)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right", fontsize=8)

    fig.suptitle("Socioeconomic Indicators vs Top-2 Correlated Emotion Counts (z-score normalized)",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "socioeco_vs_emotion.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] socioeco_vs_emotion.png")


# ── 6. 자살자수 ↔ 감정 lagged cross-correlation ────────────────────────────
def plot_lagged_corr(df, save_dir, max_lag=6):
    """emotion[t] vs suicide[t+lag] 상관 (lag=0..max_lag)"""
    count_cols = [f"cnt_{en}" for en in EMOTION_EN]
    lags = range(0, max_lag + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, en, color in zip(count_cols, EMOTION_EN, EMOTION_COLORS):
        rs = []
        for lag in lags:
            emo = df[col].values[: len(df) - lag]
            suc = df["자살사망자수"].values[lag:]
            r   = np.corrcoef(emo, suc)[0, 1]
            rs.append(r)
        ax.plot(list(lags), rs, marker="o", ms=6, lw=2, color=color, label=en)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Lag (months):  emotion[t] vs suicide[t + lag]")
    ax.set_ylabel("Pearson r")
    ax.set_title("Lagged Cross-Correlation: Emotion Count → Future Suicide Deaths", fontsize=12)
    ax.set_xticks(list(lags))
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "corr_suicide_emotion.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] corr_suicide_emotion.png")


# ── 7. 사회경제 ↔ 감정 상관 히트맵 ──────────────────────────────────────────
def plot_corr_heatmap(df, save_dir):
    try:
        import seaborn as sns
    except ImportError:
        print("  [SKIP] seaborn not installed, skipping corr_heatmap.png")
        return

    count_cols = [f"cnt_{en}" for en in EMOTION_EN]
    socio_cols = [c for c in SOCIOECO if c in df.columns]

    # 사회경제 지표 ↔ 감정 raw count 상관
    sub = df[socio_cols + count_cols].copy()
    sub.columns = [SOCIOECO.get(c, c) for c in socio_cols] + EMOTION_EN
    corr = sub.corr().loc[list(SOCIOECO.values()), EMOTION_EN]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
        vmin=-1, vmax=1
    )
    ax.set_title("Correlation: Socioeconomic Indicators × Emotion Counts", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "corr_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] corr_heatmap.png")

    # 자살자수 ↔ 감정 상관도 출력
    r_suicide = df[count_cols + ["자살사망자수"]].corr().loc["자살사망자수", count_cols]
    r_suicide.index = EMOTION_EN
    print("\n  [Suicide ↔ Emotion raw count correlation]")
    print(r_suicide.sort_values(ascending=False).to_string())


# ── main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    df = load_data(CSV_PATH)

    print(f"\nData: {len(df)} months  ({df['날짜'].min().date()} ~ {df['날짜'].max().date()})")
    print(f"Suicide range: {df['자살사망자수'].min()} ~ {df['자살사망자수'].max()}")
    print(f"Total comments range: {df['댓글수'].min()} ~ {df['댓글수'].max()}")
    print(f"\nSaving EDA plots to: {SAVE_DIR}/\n")

    plot_suicide_trend(df, SAVE_DIR)
    plot_emotion_counts(df, SAVE_DIR)
    plot_emotion_ratios(df, SAVE_DIR)
    plot_suicide_vs_emotion(df, SAVE_DIR)
    plot_socioeco_vs_emotion(df, SAVE_DIR)
    plot_lagged_corr(df, SAVE_DIR)
    plot_corr_heatmap(df, SAVE_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
