# --- Re-import necessary libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Define Transaction Interference Model ---
def transaction_interference_model(t, rho_T0, lambda_c, alpha):
    """
    Models the evolution of transaction density over time with an interference term.
    
    Parameters:
    - t: Time variable
    - rho_T0: Initial transaction density
    - lambda_c: Interaction coefficient governing transaction growth
    - alpha: Interference strength (modulation amplitude)

    Returns:
    - rho_T: Transaction density as a function of time
    """
    return rho_T0 * np.exp(lambda_c * t) * (1 + alpha * np.sin(2 * np.pi * t / 10))

# --- Generate Simulated Data ---
time = np.linspace(0, 10, 100)  # Time range from 0 to 10
rho_T0 = 0.5  # Initial transaction density
lambda_c = 0.1  # Interaction coefficient for transaction evolution
alpha = 0.2  # Strength of interference (oscillatory component)

# Compute transaction density over time
rho_T_values = transaction_interference_model(time, rho_T0, lambda_c, alpha)

# --- Create a DataFrame for the results ---
df_interference = pd.DataFrame({
    "Time (t)": time,
    "Transaction Density (ρ_T)": rho_T_values
})

# --- Plot the Transaction Density Evolution ---
plt.figure(figsize=(8,5))
plt.plot(time, rho_T_values, 
         label=r"$\rho_T(t) = \rho_{T0} e^{\lambda_c t} (1 + \alpha \sin(2\pi t / 10))$", 
         color="b")
plt.xlabel("Time (t)")
plt.ylabel("Transaction Density (ρ_T)")
plt.title("Transaction Interference Effect on Density")
plt.legend()
plt.grid(True)
plt.show()

# --- Display the DataFrame (For Non-Colab Environments, Use print Instead) ---
print("Transaction Interference Simulation Results:")
print(df_interference.head(10))  # Display first 10 rows
