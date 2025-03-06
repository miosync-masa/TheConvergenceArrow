import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 既存のQZEデータ（観測頻度 vs. 生存確率）
obs_freq = np.array([1, 5, 10, 20, 50])  # Hz
survival_prob = np.array([0.95, 0.85, 0.75, 0.60, 0.40])  # 生存確率

# Shannon エントロピー計算
entropy_values = -survival_prob * np.log(survival_prob)

# λc 依存モデル (収束の矢のエントロピー増加率)
def entropy_lambda_model(f, S0, lambda_c, alpha):
    return S0 / (1 + lambda_c * np.log(1 + alpha * f))

# 初期推定値を変更
initial_guess_lambda = [max(entropy_values), 0.01, 2]  # S0, lambda_c, alpha

# λc 依存モデルのフィッティング
params_lambda, _ = curve_fit(entropy_lambda_model, obs_freq, entropy_values, p0=initial_guess_lambda, maxfev=5000)

# フィッティング結果を取得
S0_lambda, lambda_c_fit, alpha_fit = params_lambda
print(f"Fitted Parameters: S0={S0_lambda}, λc={lambda_c_fit}, α={alpha_fit}")

# さまざまな λc の値を設定
lambda_c_values = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

# 周波数の範囲を拡張
freq_range = np.linspace(1, 50, 100)

plt.figure(figsize=(8, 5))

# 各 λc に対してエントロピー変化をプロット
for lambda_c in lambda_c_values:
    entropy_curve = entropy_lambda_model(freq_range, S0_lambda, lambda_c, alpha_fit)
    plt.plot(freq_range, entropy_curve, label=f"λc = {lambda_c:.4f}")

# 観測データもプロット
plt.scatter(obs_freq, entropy_values, color='red', label="Observed Data", zorder=3)

# 軸ラベルとタイトル
plt.xlabel("Observation Frequency (Hz)")
plt.ylabel("Entropy (Shannon Approximation)")
plt.title("Entropy Growth vs. λc Variation (Improved)")
plt.legend()
plt.grid()

# グラフを表示
plt.show()
