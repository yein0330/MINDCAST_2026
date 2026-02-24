"""모델 모듈 — 2단계 자살 예측 모델

이론적 프레임워크:
  가정 1: Macro stress → Emotional amplification  (Stage 1: EmotionAmplifier)
  가정 2: Emotional amplification → Suicide variation (Stage 2: SuicideForecaster)

  전제 1: 뉴스 데이터는 사회·경제·정치적 변동(사회적 이슈)에 대한 사건을 다룬다.
  전제 2: 뉴스 댓글의 감정은 사회적 이슈에 대한 반응·결과이다.
"""
import math
import torch
import torch.nn as nn

EMOTION_NAMES = ["분노", "슬픔", "불안", "상처", "당황", "기쁨"]


class SinusoidalPositionalEncoding(nn.Module):
    """사인-코사인 위치 인코딩"""

    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        S = x.size(1)
        if S > self.pe.size(0):
            pe = torch.zeros(S, self.d_model, device=x.device)
            pos = torch.arange(0, S, device=x.device).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, self.d_model, 2, device=x.device)
                * (-math.log(10000.0) / self.d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.pe = pe
        return x + self.pe[:S]


class EmotionAmplifier(nn.Module):
    """Stage 1: 사회경제적 스트레스 시계열 → 감정 증폭 벡터

    6개의 감정 쿼리(분노·슬픔·불안·상처·당황·기쁨)가
    스트레스 지표 시퀀스에 Cross-Attention하여
    각 감정의 증폭 강도를 스칼라로 출력.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_emotions: int = 6,
        max_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model, max_len)
        self.norm_kv = nn.LayerNorm(d_model)

        # 감정별 학습 가능한 쿼리 벡터
        self.emotion_queries = nn.Parameter(
            torch.randn(1, n_emotions, d_model) * 0.02
        )
        self.norm_q = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # 감정 스코어 헤드: d_model → scalar per emotion
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, X, mask, return_attn: bool = False):
        """
        Args:
            X    : (B, T, d_in)  스트레스 지표 시계열
            mask : (B, T)        패딩 마스크 (True = 패딩)
        Returns:
            emotion_vec : (B, n_emotions)    감정 증폭 벡터
            attn        : (B, n_emotions, T) [return_attn=True 일 때만]
        """
        B = X.size(0)
        kv = self.norm_kv(self.posenc(self.proj(X)))              # (B, T, d_model)
        q = self.norm_q(self.emotion_queries.expand(B, -1, -1))   # (B, n_emotions, d_model)

        out, attn = self.cross_attn(
            q, kv, kv, key_padding_mask=mask
        )  # out: (B, n_emotions, d_model), attn: (B, n_emotions, T)

        emotion_vec = self.score_head(out).squeeze(-1)  # (B, n_emotions)

        if return_attn:
            return emotion_vec, attn
        return emotion_vec


class SuicideForecaster(nn.Module):
    """Stage 2: 감정 증폭 벡터 → 자살자 수 예측

    n_out 개의 독립 예측 헤드 사용.
    - n_out=1 : 자살자수 합계 단일 예측
    - n_out=2 : 남자/여자 개별 예측 (성별 데이터가 있을 때)
    """

    def __init__(self, n_emotions: int = 6, d_hidden: int = 128, n_out: int = 2):
        super().__init__()
        self.n_out = n_out
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_emotions, d_hidden),
                nn.ReLU(),
                nn.LayerNorm(d_hidden),
                nn.Linear(d_hidden, 1),
            )
            for _ in range(n_out)
        ])

    def forward(self, emotion_vec):
        """
        Args:
            emotion_vec : (B, n_emotions)
        Returns:
            y_hat : (B, n_out)
        """
        return torch.cat([h(emotion_vec) for h in self.heads], dim=-1)


class EmotionSequenceForecaster(nn.Module):
    """Stage 2: emotion distribution sequence → suicide count.

    Stage 1의 EmotionAmplifier와 대칭 구조:
      EmotionAmplifier        : stress_seq   → Cross-Attn(emotion queries) → emotion_vec
      EmotionSequenceForecaster: emotion_seq → Cross-Attn(suicide queries) → suicide_count

    여러 시점의 감정 분포(lag sequence)를 Key/Value로 받아
    학습 가능한 n_out개 쿼리가 Cross-Attention하여 자살자 수를 예측.
    lag가 다를 때 어느 시점 감정이 자살에 영향을 주는지 attention weight로 해석 가능.
    """

    def __init__(
        self,
        n_emotions: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        n_out: int = 1,
        max_lag: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(n_emotions, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model, max_lag)
        self.norm_kv = nn.LayerNorm(d_model)

        # n_out 개의 학습 가능한 쿼리 (예측 대상별)
        self.out_queries = nn.Parameter(torch.randn(1, n_out, d_model) * 0.02)
        self.norm_q = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, emotion_seq, mask=None, return_attn: bool = False):
        """
        Args:
            emotion_seq : (B, L, n_emotions)  감정 시계열
            mask        : (B, L)              True = 패딩
        Returns:
            y_hat : (B, n_out)
            attn  : (B, n_out, L)  [return_attn=True 일 때만]
        """
        B = emotion_seq.size(0)
        kv = self.norm_kv(self.posenc(self.proj(emotion_seq)))  # (B, L, d_model)
        q  = self.norm_q(self.out_queries.expand(B, -1, -1))    # (B, n_out, d_model)

        out, attn = self.cross_attn(q, kv, kv, key_padding_mask=mask)
        y_hat = self.score_head(out).squeeze(-1)  # (B, n_out)

        if return_attn:
            return y_hat, attn
        return y_hat


class TwoStageSuicideModel(nn.Module):
    """2단계 자살 예측 모델

    가정 1: Macro stress → Emotional amplification  (stage1: EmotionAmplifier)
    가정 2: Emotional amplification → Suicide variation (stage2: SuicideForecaster)

    자살자 수 MSELoss 단일 지도 신호로 두 스테이지를 동시에 end-to-end 학습.
    Stage 1의 감정 벡터는 명시적 감정 레이블 없이 자살 예측에 최적화된
    잠재 감정 표현을 자동으로 학습.
    """

    EMOTION_NAMES = EMOTION_NAMES

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_emotions: int = 6,
        max_len: int = 32,
        dropout: float = 0.1,
        d_hidden: int = 128,
        n_out: int = 2,
    ):
        super().__init__()
        self.stage1 = EmotionAmplifier(
            d_in, d_model, n_heads, n_emotions, max_len, dropout
        )
        self.stage2 = SuicideForecaster(n_emotions, d_hidden, n_out)

    def forward(self, X, mask, return_emotion: bool = False, return_attn: bool = False):
        """
        Args:
            X             : (B, T, d_in)
            mask          : (B, T)
            return_emotion: True → (y_hat, emotion_vec) 반환
            return_attn   : True → (y_hat, attn) 반환
        Returns:
            y_hat       : (B, 2)               [남자, 여자]
            emotion_vec : (B, n_emotions)      [return_emotion=True]
            attn        : (B, 1, n_emotions, T) [return_attn=True]
        """
        if return_attn:
            emotion_vec, attn = self.stage1(X, mask, return_attn=True)
        else:
            emotion_vec = self.stage1(X, mask)

        y_hat = self.stage2(emotion_vec)

        if return_attn:
            if attn.ndim == 3:
                attn = attn.unsqueeze(1)
            return y_hat, attn

        if return_emotion:
            return y_hat, emotion_vec

        return y_hat
