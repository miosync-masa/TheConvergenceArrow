import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# 提供されたデータ（λc vs. ブラックホール蒸発時間）
lambda_c_values = np.array([0.0, 0.1, 0.3, 0.5, 1.0])
T_evap_values = np.array([5.0, 5.5, 7.0, 9.0, 15.0])  # 例: 収束の矢モデルによる蒸発時間

# 改良モデル
def evap_time_model(lambda_c, T0, alpha):
    return T0 * np.exp(alpha * lambda_c)

# フィッティングの実行
initial_guess = [5.0, 1.0]  # 初期推定値（T0, α）
params, covariance = curve_fit(evap_time_model, lambda_c_values, T_evap_values, p0=initial_guess)

# フィッティング結果の取得
T0_fit, alpha_fit = params
print(f"\nFitted Parameters:")
print(f"Estimated T0 = {T0_fit:.5f}")
print(f"Estimated α = {alpha_fit:.5f}")

# フィッティングモデルのプロット
lambda_c_range = np.linspace(0, 1, 100)
T_evap_fit = evap_time_model(lambda_c_range, T0_fit, alpha_fit)

plt.figure(figsize=(8, 5))
plt.scatter(lambda_c_values, T_evap_values, color='red', label="Observed Data")
plt.plot(lambda_c_range, T_evap_fit, 'b-', label="Fitted Model")
plt.xlabel("Information Convergence Rate (λc)")
plt.ylabel("Black Hole Evaporation Time T_evap")
plt.title("Fitting of Black Hole Evaporation Model with Information Convergence")
plt.legend()
plt.grid()
plt.show()
