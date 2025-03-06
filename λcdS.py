import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# 既存のデータ（λc vs dS/dλc）
lambda_c_fit_values = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
dS_dlambda_c_values = np.array([-0.527718, -0.492488, -0.435971, -0.362055, -0.239899, -0.126113])

# 理論モデルの定義
def theoretical_entropy_grad(lambda_c, S0, alpha):
    return -S0 / ((1 + lambda_c) ** 2) * np.exp(-alpha * lambda_c)

# 初期推定値
initial_guess = [max(abs(dS_dlambda_c_values)), 0.1]  # S0, α

# フィッティング実行
params_theory, _ = curve_fit(theoretical_entropy_grad, lambda_c_fit_values, dS_dlambda_c_values, p0=initial_guess)

# フィッティング結果取得
S0_fit, alpha_fit = params_theory

# 数値データ出力
print(f"\nFitted Parameters:")
print(f"S₀ = {S0_fit:.5f}")
print(f"α  = {alpha_fit:.5f}")

# λc の範囲を広げて理論モデルをプロット
lambda_c_range = np.linspace(0.0001, 0.1, 100)
grad_fit = theoretical_entropy_grad(lambda_c_range, S0_fit, alpha_fit)

# 実測データと比較プロット
plt.figure(figsize=(8, 5))
plt.scatter(lambda_c_fit_values, dS_dlambda_c_values, color='red', label="Observed Data")
plt.plot(lambda_c_range, grad_fit, 'b-', label="Fitted Model")
plt.xlabel("λc")
plt.ylabel("dS/dλc")
plt.title("Comparison of Theoretical Model with Observed Data")
plt.legend()
plt.grid()

# グラフを表示
plt.show()

# フィッティング結果をデータフレーム化
df_fit_results = pd.DataFrame({
    "λc": lambda_c_fit_values,
    "Observed dS/dλc": dS_dlambda_c_values,
    "Fitted dS/dλc": theoretical_entropy_grad(lambda_c_fit_values, S0_fit, alpha_fit)
})

# 数値データの出力
print("\nComparison of Observed and Fitted dS/dλc Values:")
print(df_fit_results)
