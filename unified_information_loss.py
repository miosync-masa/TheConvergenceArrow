import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 時間ステップ
lambda_c_values = np.linspace(0, 1, 100)

# 統一情報流出モデル（ブラックホール & エンタングルメント崩壊）
def unified_information_loss(lambda_c, S0, alpha):
    return -S0 / ((1 + lambda_c)**2) * np.exp(-alpha * lambda_c)

# 初期エントロピー
S0_blackhole = 1.0  # ブラックホール情報流出の基準値
S0_entanglement = 1.0  # エンタングルメント崩壊の基準値

# 減衰パラメータ
alpha_blackhole = 0.5
alpha_entanglement = 0.3

# ブラックホール情報流出のシミュレーション
S_blackhole = unified_information_loss(lambda_c_values, S0_blackhole, alpha_blackhole)

# エンタングルメント崩壊のシミュレーション
S_entanglement = unified_information_loss(lambda_c_values, S0_entanglement, alpha_entanglement)

# グラフプロット
plt.figure(figsize=(8, 5))
plt.plot(lambda_c_values, S_blackhole, label="Black Hole Information Loss", color='blue')
plt.plot(lambda_c_values, S_entanglement, label="Entanglement Collapse", color='red')
plt.xlabel("Information Convergence Rate (λc)")
plt.ylabel("Normalized Information Loss")
plt.title("Unified Model: Black Hole Information vs. Entanglement Collapse")
plt.legend()
plt.grid()
plt.show()

# データ出力
df_unified = pd.DataFrame({
    "Lambda_c": lambda_c_values,
    "Black Hole Information Loss": S_blackhole,
    "Entanglement Collapse": S_entanglement
})

print("\nSimulated Unified Information Loss Data:")
print(df_unified.head(10))  # 最初の10行のみ表示
