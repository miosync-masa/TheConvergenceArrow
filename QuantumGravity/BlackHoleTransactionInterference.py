# --- Required Libraries ---
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Black Hole Entropy Growth Model with Transaction Interference ---
def S_BH_interference(t, S0, lambda_c, rho_T0, alpha):
    """
    Computes black hole entropy evolution considering transaction interference.

    Parameters:
    - t: Time variable
    - S0: Maximum entropy (final entropy state)
    - lambda_c: Convergence rate of information transactions
    - rho_T0: Initial transaction density
    - alpha: Interference factor (modulates transaction density)

    Returns:
    - S_BH: Black hole entropy as a function of time
    """
    # Transaction density model incorporating interference effects
    rho_T = rho_T0 * np.exp(lambda_c * t) * (1 + alpha * np.sin(2 * np.pi * t / 10))
    
    # Prevent numerical overflow by limiting exponent range
    exponent = -lambda_c * rho_T * t
    exponent = np.clip(exponent, -700, 700)  # Clipping to avoid overflow
    return S0 * (1 - np.exp(exponent))

# --- Generate Simulated Data ---
t_values = np.linspace(0, 10, 100)  # Time range
S0 = 1.0  # Maximum entropy
lambda_c = 0.1  # Interaction coefficient for transactions
rho_T0 = 0.5  # Initial transaction density
alpha_values = [0.0, 0.2, -0.2]  # No interference, Amplification, Suppression

# Compute black hole entropy for different interference levels
S_BH_results = {alpha: S_BH_interference(t_values, S0, lambda_c, rho_T0, alpha) for alpha in alpha_values}

# --- Plot the Results ---
plt.figure(figsize=(8, 5))
for alpha, S_BH in S_BH_results.items():
    plt.plot(t_values, S_BH, label=f"α = {alpha:.1f}")

plt.xlabel("Time (t)")
plt.ylabel("Black Hole Entropy $S_{BH}$")
plt.title("Black Hole Information Loss with Transaction Interference")
plt.legend()
plt.grid(True)
plt.show()

# --- Create DataFrame for Results ---
df_S_BH_interference = pd.DataFrame({
    "Time (t)": t_values,
    "S_BH (No Interference)": S_BH_results[0.0],
    "S_BH (Amplification)": S_BH_results[0.2],
    "S_BH (Suppression)": S_BH_results[-0.2]
})

# --- Display Optimized Data (For Non-Colab Environments, Use print Instead) ---
print("\nSimulated Data for Black Hole Entropy with Transaction Interference:")
print(df_S_BH_interference.head(10))  # Display first 10 rows
