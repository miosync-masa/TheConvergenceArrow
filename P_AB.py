import numpy as np
import pandas as pd

# 量子ゼノ効果データ（観測頻度 vs. 生存確率）
obs_freq = np.array([1, 5, 10, 20, 50])  # Hz
survival_prob = np.array([0.95, 0.85, 0.75, 0.60, 0.40])  # 生存確率

# 条件付き確率 P(A|B) を仮定（未来の情報収束が強いほど変化する）
# ここでは、P(A|B) を生存確率 P(A) と測定頻度の関数としてモデル化
P_A_given_B = survival_prob * np.exp(-obs_freq / 50)

# 情報エントロピー S(A), S(B), S(A|B) を計算
S_A = -survival_prob * np.log(survival_prob)  # 未来状態 A のエントロピー
S_B = -(1 - survival_prob) * np.log(1 - survival_prob)  # 現在状態 B のエントロピー
S_A_given_B = -P_A_given_B * np.log(P_A_given_B)  # 条件付きエントロピー S(A|B)

# 相関エントロピー S(A,B) = S(A|B) + S(B)
S_AB = S_A_given_B + S_B

# 相互情報量 I(A;B) を計算
I_AB = S_A + S_B - S_AB

# データフレームを作成
df_mutual_info_corrected = pd.DataFrame({
    "Observation Frequency (Hz)": obs_freq,
    "Survival Probability": survival_prob,
    "Entropy S(A)": S_A,
    "Entropy S(B)": S_B,
    "Conditional Entropy S(A|B)": S_A_given_B,
    "Joint Entropy S(A,B)": S_AB,
    "Mutual Information I(A;B)": I_AB
})

# 数値データを出力
print("\nCorrected Mutual Information Calculation:")
print(df_mutual_info_corrected)

