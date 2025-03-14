# ---BlackHoleInformationLossModel with Transaction Density Interference---

import numpy as np
import matplotlib.pyplot as plt

# --- Parameter Setup ---
lambda_c = 0.1  # Transaction influence coefficient
rho_T_0 = 0.5  # Initial transaction density
time = np.linspace(0, 10, 100)  # Time range for simulation

# --- Transaction Density Evolution with Interference ---
# Exponential decay model with additional interference term
rho_T_interference = rho_T_0 * np.exp(-lambda_c * time) + lambda_c * np.cumsum(np.exp(-lambda_c * time))

# --- Black Hole Information Loss Model with Transaction Density ---
# Modified Page curve incorporating transaction density dynamics
S_BH = (1 - np.exp(-lambda_c * rho_T_interference * time))

# --- Plot the Results ---
plt.figure(figsize=(8, 5))
plt.plot(time, S_BH, label=r"$S_{BH}(t) = S_0(1 - e^{-\lambda_c \rho_T t})$", color='blue')
plt.xlabel("Time (t)")
plt.ylabel("Black Hole Entropy $S_{BH}$")
plt.title("Black Hole Information Loss Model with Transaction Density")
plt.legend()
plt.grid()
plt.show()
