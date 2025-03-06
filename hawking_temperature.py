import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# パラメータ設定
M_values = np.linspace(1, 10, 100)  # ブラックホールの質量範囲
E_values = np.linspace(0, 1, 100)  # エンタングルメント強度

# ホーキング温度の計算
def hawking_temperature(M):
    return 1 / (8 * np.pi * M)

# 修正された情報流出モデル
def information_loss(M, E, alpha):
    T_H = hawking_temperature(M)
    return -1 / M**2 * np.exp(-alpha / T_H) * np.exp(-alpha * E)

# 計算
loss_data = np.array([[information_loss(M, E, 0.5) for E in E_values] for M in M_values])

# データフレーム作成
df_info_loss = pd.DataFrame(loss_data, index=M_values, columns=E_values)

# グラフのプロット
plt.figure(figsize=(8, 5))
plt.contourf(E_values, M_values, loss_data, levels=20, cmap="inferno")
plt.colorbar(label="Information Loss Rate")
plt.xlabel("Entanglement Strength E")
plt.ylabel("Black Hole Mass M")
plt.title("Information Loss vs. Black Hole Mass and Entanglement Strength")
plt.show()

# 数値データを表示（最初の10行のみ）
print("\nSimulated Information Loss Data:")
print(df_info_loss.iloc[:10, :10])  # 最初の10x10ブロックを表示
