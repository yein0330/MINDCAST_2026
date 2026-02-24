"""유틸리티 함수 모듈 — 시드, 폰트, CSV 로드

matplotlib / seaborn / torch는 실제로 필요한 함수 내부에서만 임포트.
→ data_loader 단독 실행 시 불필요한 패키지 의존성을 피함.
"""
import os
import random
import numpy as np
import pandas as pd

# -----------------------------
# Config (기본값)
# -----------------------------
CSV_PATH = "/home/yein38/mindcastlib_trainer/data/base/base_data.csv"
SAVE_ROOT = "results/emotion_stage"
TARGET_DATE = "2023-04-01"
MAX_LEN = 32
BATCH_SIZE = 64
EPOCHS = 200
LR = 2e-4
SEED = 42


def set_seed(seed=42):
    """재현성을 위한 시드 고정"""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_korean_font():
    """한글 폰트 자동 설정"""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.rcParams["axes.unicode_minus"] = False
    candidates = [
        # matplotlib 사용자 폰트 디렉터리 (다운로드된 폰트)
        os.path.join(matplotlib.get_configdir(), "fonts", "NanumGothic-Regular.ttf"),
        # 시스템 설치 폰트
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            plt.rcParams["font.family"] = name
            print(f"Using font: {name}")
            return
    print("Korean font not found; fallback to default.")


def load_clean_csv(path: str) -> pd.DataFrame:
    """CSV를 읽을 때 Unnamed 컬럼 자동 제거"""
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df
