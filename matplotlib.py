import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 情報収束率 λc の範囲
lambda_c_values = np.linspace(0, 5, 100)

# 時間の新しい定義
def new_time_definition(lambda_c, T0, beta):
    return T0 * np.exp(beta * lambda_c)

# パラメータ設定
T0 = 1.0  # 従来の時間基準
beta = 1.1  # 収束率の影響の強さ

# 時間の計算
t_values = new_time_definition(lambda_c_values, T0, beta)

# グラフのプロット
plt.figure(figsize=(8, 5))
plt.plot(lambda_c_values, t_values, label="New Time Definition")
plt.xlabel("Information Convergence Rate (λc)")
plt.ylabel("Time t(λc)")
plt.title("Time as a Function of Information Convergence Rate")
plt.legend()
plt.grid()
plt.show()

# 数値データを出力
df_time = pd.DataFrame({
    "Lambda_c": lambda_c_values,
    "Time t(λc)": t_values
})

print("\nSimulated Time Function Data:")
print(df_time.head(10))  # 最初の10行のみ表示
