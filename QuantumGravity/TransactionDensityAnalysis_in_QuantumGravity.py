# TransactionDensityAnalysis_in_QuantumGravity

import numpy as np
import matplotlib.pyplot as plt

# --- Parameter Setup ---
E_values = np.logspace(-2, 5, 200)  # Energy scale in eV (logarithmic range)
hbar = 6.582e-16  # Reduced Planck's constant (eV⋅s)
lambda_c = 0.1  # Information leakage coefficient
N_T = 1.0  # Number of transactions (unitless)

# --- 1. Transaction Density Evolution in Quantum Gravity ---
rho_T = np.clip(E_values / hbar, 1e-10, None)  # Normalized transaction density
rho_T_modulated = rho_T * (1 + 0.1 * np.sin(2 * np.pi * E_values / 10))  # Small oscillatory correction

# --- 2. Black Hole Information Flow under Transaction Density ---
S_BH = 1 - np.exp(-lambda_c * N_T / rho_T)  # Black hole entropy due to information leakage

# --- 3. Early Universe: Transaction Density Growth ---
rho_T_bigbang = np.exp(-1 / np.clip((E_values / hbar), 1e-10, None))  # Stabilized growth in early universe

# --- Plot the Results ---
fig, axs = plt.subplots(3, 1, figsize=(8, 14))

# 1. Transaction Density Evolution in Quantum Gravity
axs[0].set_title("Transaction Density Evolution in Quantum Gravity")
axs[0].plot(E_values, rho_T, 'b', label=r"$\rho_T = E / \hbar$")
axs[0].plot(E_values, rho_T_modulated, 'g--', label=r"Modulated $\rho_T$")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel("Energy Scale (E) [eV]")
axs[0].set_ylabel("Transaction Density $\\rho_T$")
axs[0].legend()
axs[0].grid(True, which="both")

# 2. Black Hole Information Flow under Transaction Density
axs[1].set_title("Black Hole Information Flow under Transaction Density")
axs[1].plot(E_values, S_BH, 'r', label=r"$S_{BH} = 1 - e^{-\lambda_c N_T / \rho_T}$")
axs[1].set_xscale("log")
axs[1].set_xlabel("Energy Scale (E) [eV]")
axs[1].set_ylabel("Entropy $S_{BH}$")
axs[1].legend()
axs[1].grid(True, which="both")

# 3. Early Universe: Transaction Density Growth
axs[2].set_title("Big Bang as Transaction Density Growth")
axs[2].plot(E_values, rho_T_bigbang, 'm', label=r"$\rho_T \propto e^{-1 / (E / \hbar)}$")
axs[2].set_xscale("log")
axs[2].set_xlabel("Energy Scale (E) [eV]")
axs[2].set_ylabel("Big Bang Transaction Density")
axs[2].legend()
axs[2].grid(True, which="both")

plt.tight_layout()
plt.show()
