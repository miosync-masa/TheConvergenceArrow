import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ブラックホール蒸発のシミュレーションパラメータ
time_steps = np.linspace(0, 10, 100)  # 時間の正規化 (T/T_Page)
lambda_c_values = [0.0, 0.1, 0.3, 0.5]  # 収束率 λc の異なるケース

# 従来のホーキング放射モデル（情報収束なし）
def standard_evaporation_rate(t, M0=1):
    return M0 / (1 + t)

# 修正ホーキング放射モデル（収束の矢理論）
def modified_evaporation_rate(t, lambda_c, M0=1):
    return M0 / (1 + (1 + lambda_c) * t)

# グラフのプロット
plt.figure(figsize=(8, 5))

# 各 λc に対する修正ホーキング放射曲線
for lambda_c in lambda_c_values:
    plt.plot(time_steps, modified_evaporation_rate(time_steps, lambda_c), label=f"λc = {lambda_c}")

# 従来のホーキング放射曲線
plt.plot(time_steps, standard_evaporation_rate(time_steps), 'k--', label="Standard Hawking Radiation")

# ラベルとタイトル
plt.xlabel("Normalized Time (T/T_Page)")
plt.ylabel("Black Hole Mass M(T)")
plt.title("Black Hole Evaporation with Information Convergence")
plt.legend()
plt.grid()

# グラフを表示
plt.show()

# 数値データの出力
df_evaporation = pd.DataFrame({
    "Time (T/T_Page)": time_steps,
    "Standard Evaporation": standard_evaporation_rate(time_steps),
})

# 各 λc の蒸発データを追加
for lambda_c in lambda_c_values:
    df_evaporation[f"Modified Evaporation (λc={lambda_c})"] = modified_evaporation_rate(time_steps, lambda_c)

# 数値データの表示
print("\nSimulated Black Hole Evaporation Data:")
print(df_evaporation.head(10))  # 最初の10行のみ表示
