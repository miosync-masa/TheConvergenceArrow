import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# ブラックホールの初期質量（太陽質量単位）
M_initial = 5  # 例えば小型ブラックホール
time_steps = np.linspace(0, 10, 500)  # 正規化時間 T/T_Page
lambda_c_values = [0.0, 0.1, 0.3, 0.5, 1.0]  # 収束率 λc の異なるケース

# 従来のホーキング放射による蒸発時間スケール
def standard_evaporation_time(M0):
    return M0**3  # Hawking蒸発スケール ~ M^3

# 収束の矢モデルに基づく修正蒸発時間スケール
def modified_evaporation_time(M0, lambda_c):
    return (1 + lambda_c) * M0**3  # 情報収束の影響を考慮

# 収束の矢理論を含む新しい蒸発モデル（フィッティング用）
def evaporation_model(t, M0, lambda_c):
    return M0 / (1 + lambda_c * t)  # 情報収束を考慮したモデル

# グラフのプロット
plt.figure(figsize=(8, 5))

# 各 λc に対するブラックホール蒸発の時間スケール
for lambda_c in lambda_c_values:
    lifetime = modified_evaporation_time(M_initial, lambda_c)
    plt.plot(time_steps, evaporation_model(time_steps, M_initial, lambda_c), label=f"λc = {lambda_c}")

# 従来のホーキング放射曲線
plt.plot(time_steps, M_initial / (1 + time_steps), 'k--', label="Standard Hawking Radiation")

# ラベルとタイトル
plt.xlabel("Normalized Time (T/T_Page)")
plt.ylabel("Black Hole Mass M(T)")
plt.title("Black Hole Evaporation Time Scale with Information Convergence")
plt.legend()
plt.grid()

# グラフを表示
plt.show()

# 数値データの出力
df_evaporation_time = pd.DataFrame({
    "Time (T/T_Page)": time_steps,
    "Standard Evaporation": M_initial / (1 + time_steps)
})

# 各 λc の蒸発時間データを追加
for lambda_c in lambda_c_values:
    df_evaporation_time[f"Modified Evaporation (λc={lambda_c})"] = evaporation_model(time_steps, M_initial, lambda_c)

# 数値データの表示
print("\nSimulated Black Hole Evaporation Time Scale Data:")
print(df_evaporation_time.head(10))  # 最初の10行のみ表示

# -------------------------------------------------------------
# フィッティングの実行
# -------------------------------------------------------------

# 実測データの作成（ここではシミュレーションデータを基準に）
observed_time = time_steps
observed_mass = M_initial / (1 + 0.2 * time_steps)  # 仮に λc = 0.2 のケースを実測データとする

# フィッティングの実行
params, covariance = curve_fit(evaporation_model, observed_time, observed_mass, p0=[M_initial, 0.1])

# フィッティング結果の取得
M0_fit, lambda_c_fit = params
print(f"\nFitted Parameters:")
print(f"Estimated M0 = {M0_fit:.5f}")
print(f"Estimated λc = {lambda_c_fit:.5f}")

# フィッティングモデルのプロット
fitted_mass = evaporation_model(time_steps, M0_fit, lambda_c_fit)

plt.figure(figsize=(8, 5))
plt.scatter(observed_time, observed_mass, color='red', label="Observed Data")
plt.plot(time_steps, fitted_mass, 'b-', label="Fitted Model")
plt.xlabel("Normalized Time (T/T_Page)")
plt.ylabel("Black Hole Mass M(T)")
plt.title("Fitting of Black Hole Evaporation Model")
plt.legend()
plt.grid()
plt.show()
