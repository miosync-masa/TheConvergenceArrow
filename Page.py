import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ブラックホールのPage曲線の数値シミュレーション

# シミュレーション用パラメータ
time_steps = np.linspace(0, 1, 100)  # 正規化した時間 (t/T_Page)
lambda_c_values = [0.0, 0.1, 0.3, 0.5]  # 収束率 λc の異なるケース

# 標準のPage曲線モデル（情報回収なしの従来モデル）
def standard_page_curve(t):
    return 2 * t / (1 + t)

# 修正Page曲線モデル（収束の矢理論）
def modified_page_curve(t, lambda_c):
    return (1 / (1 + lambda_c)) * standard_page_curve(t)

# グラフのプロット
plt.figure(figsize=(8, 5))

# 収束の矢モデルによる曲線
for lambda_c in lambda_c_values:
    plt.plot(time_steps, modified_page_curve(time_steps, lambda_c), label=f"λc = {lambda_c}")

# 従来のPage曲線をプロット
plt.plot(time_steps, standard_page_curve(time_steps), 'k--', label="Standard Page Curve")

# ラベルとタイトル
plt.xlabel("Normalized Time (t/T_Page)")
plt.ylabel("Entropy S")
plt.title("Black Hole Page Curve with Information Convergence")
plt.legend()
plt.grid()

# グラフを表示
plt.show()

# 数値データの出力
df_page_curve = pd.DataFrame({
    "Time (t/T_Page)": time_steps,
    "Standard Page Curve": standard_page_curve(time_steps),
})

# 各 λc のPage曲線データを追加
for lambda_c in lambda_c_values:
    df_page_curve[f"Modified Page Curve (λc={lambda_c})"] = modified_page_curve(time_steps, lambda_c)

# 数値データの表示
print("\nSimulated Page Curve Data:")
print(df_page_curve.head(10))  # 最初の10行のみ表示

# -------------------------------------------------------------
# I(A;B) の増加とブラックホール情報回収の相関を解析
# -------------------------------------------------------------

# 相互情報量の計算モデル (仮定: 情報回収が進むと I(A;B) が増加)
def mutual_information_black_hole(t, lambda_c):
    return (1 - np.exp(-lambda_c * t)) * standard_page_curve(t)

# グラフのプロット
plt.figure(figsize=(8, 5))

for lambda_c in lambda_c_values:
    plt.plot(time_steps, mutual_information_black_hole(time_steps, lambda_c), label=f"λc = {lambda_c}")

plt.xlabel("Normalized Time (t/T_Page)")
plt.ylabel("Mutual Information I(A;B)")
plt.title("Mutual Information and Black Hole Information Recovery")
plt.legend()
plt.grid()

# グラフを表示
plt.show()

# 数値データの出力
df_mutual_info_bh = pd.DataFrame({
    "Time (t/T_Page)": time_steps,
})

# 各 λc の相互情報量データを追加
for lambda_c in lambda_c_values:
    df_mutual_info_bh[f"I(A;B) (λc={lambda_c})"] = mutual_information_black_hole(time_steps, lambda_c)

# 数値データの表示
print("\nSimulated Mutual Information I(A;B) Data:")
print(df_mutual_info_bh.head(10))  # 最初の10行のみ表示
