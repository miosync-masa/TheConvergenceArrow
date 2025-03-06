import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
print(f"Fitted Parameters: S0={S0_lambda:.5f}, λc={lambda_c_fit:.5f}, α={alpha_fit:.5f}")

# さまざまな λc の値を設定
lambda_c_values = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

# 周波数の範囲を拡張
freq_range = np.linspace(1, 50, 100)

plt.figure(figsize=(8, 5))

# データ保存用リスト
data_list = []

# 各 λc に対してエントロピー変化をプロットし、数値データを取得
for lambda_c in lambda_c_values:
    entropy_curve = entropy_lambda_model(freq_range, S0_lambda, lambda_c, alpha_fit)
    plt.plot(freq_range, entropy_curve, label=f"λc = {lambda_c:.4f}")

    # データ保存（観測周波数とエントロピー値）
    for f, e in zip(freq_range, entropy_curve):
        data_list.append([lambda_c, f, e])

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

# データフレーム作成
df_lambda_entropy = pd.DataFrame(data_list, columns=["λc", "Observation Frequency (Hz)", "Entropy"])

# 数値データを表示
print("\nEntropy Data for Different λc Values:")
print(df_lambda_entropy.head(20))  # 最初の20行のみ表示

# λc に対するエントロピー変化の勾配を計算
grad_lambda_c = []

for i in range(len(lambda_c_values) - 1):
    lambda_c1 = lambda_c_values[i]
    lambda_c2 = lambda_c_values[i + 1]
    
    # エントロピーの差分を計算（数値微分）
    entropy_diff = entropy_lambda_model(freq_range, S0_lambda, lambda_c2, alpha_fit) - \
                   entropy_lambda_model(freq_range, S0_lambda, lambda_c1, alpha_fit)
    
    # λc の変化量
    lambda_c_diff = lambda_c2 - lambda_c1
    
    # 数値的な勾配 dS/dλc を計算
    gradient = entropy_diff / lambda_c_diff
    grad_lambda_c.append([lambda_c1, lambda_c2, np.mean(gradient)])  # 平均値を格納

# 2次導関数（変曲点）の計算
second_derivative_lambda_c = []

for i in range(len(grad_lambda_c) - 1):
    lambda_c1 = grad_lambda_c[i][1]  # λc(End)
    lambda_c2 = grad_lambda_c[i + 1][1]  # 次の λc(End)
    
    # 1次導関数（勾配）の差分
    grad_diff = grad_lambda_c[i + 1][2] - grad_lambda_c[i][2]
    
    # λc の変化量
    lambda_c_diff = lambda_c2 - lambda_c1
    
    # 2次導関数 (d^2 S / dλc^2)
    second_derivative = grad_diff / lambda_c_diff
    second_derivative_lambda_c.append([lambda_c1, lambda_c2, second_derivative])

# データフレーム化して数値データを出力
df_second_deriv_lambda_c = pd.DataFrame(second_derivative_lambda_c, columns=["λc (Start)", "λc (End)", "d²S/dλc²"])
print("\nSecond Derivative of Entropy with respect to λc:")
print(df_second_deriv_lambda_c)
