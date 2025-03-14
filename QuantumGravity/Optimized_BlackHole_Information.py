# --- Required Libraries ---
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd

# --- Experimental Data (Simulated Observations) ---
t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Time points
S_BH_data = np.array([0.0000, 0.0818, 0.1487, 0.2105, 0.2676, 
                       0.3202, 0.3684, 0.4125, 0.4531, 0.4902, 0.5241])  # Observed entropy values

# --- Define the Black Hole Entropy Model ---
def S_BH_model(t, S0, lambda_c, rho_T):
    """
    Computes the evolution of black hole entropy over time.

    Parameters:
    - t: Time variable
    - S0: Maximum entropy (final entropy state)
    - lambda_c: Convergence rate of information loss
    - rho_T: Transaction density affecting entropy growth

    Returns:
    - S_BH: Black hole entropy as a function of time
    """
    # Prevent numerical overflow by limiting the exponent range
    exponent = -lambda_c * rho_T * t
    exponent = np.clip(exponent, -700, 700)  # Clipping to avoid overflow
    return S0 * (1 - np.exp(exponent))

# --- Initial Parameter Estimates ---
initial_guess = [1.0, 0.1, 0.5]  # Estimated values for S0, lambda_c, rho_T

# --- Perform Curve Fitting ---
params, covariance = opt.curve_fit(S_BH_model, t_data, S_BH_data, p0=initial_guess)

# Extract optimized parameters
S0_opt, lambda_c_opt, rho_T_opt = params

# Compute standard errors for parameter estimates
errors = np.sqrt(np.diag(covariance))
S0_err, lambda_c_err, rho_T_err = errors

# Store results in a structured dictionary
optimal_params = {
    "S0 (Maximum Entropy)": (S0_opt, S0_err),
    "lambda_c (Transaction Growth Rate)": (lambda_c_opt, lambda_c_err),
    "rho_T (Initial Transaction Density)": (rho_T_opt, rho_T_err),
}

# --- Generate Data for Fitted Curve ---
t_fit = np.linspace(0, 10, 100)  # Higher resolution time steps
S_BH_fit = S_BH_model(t_fit, *params)

# --- Plot the Results ---
plt.figure(figsize=(8, 5))
plt.scatter(t_data, S_BH_data, color='red', label="Observed Data")  # Experimental data
plt.plot(t_fit, S_BH_fit, 'b-', 
         label=f"Fitted Model: $S_{{BH}}(t) = S_0 (1 - e^{{-\\lambda_c \\rho_T t}})$\n"
               f"$S_0={S0_opt:.4f} \\pm {S0_err:.4f}$, "
               f"$\\lambda_c={lambda_c_opt:.4f} \\pm {lambda_c_err:.4f}$, "
               f"$\\rho_T={rho_T_opt:.4f} \\pm {rho_T_err:.4f}$")
plt.xlabel("Time (t)")
plt.ylabel("Black Hole Entropy $S_{BH}$")
plt.title("Optimized Black Hole Information Loss Model")
plt.legend()
plt.grid(True)
plt.show()

# --- Display Optimized Parameters ---
df_optimal_params = pd.DataFrame(optimal_params, index=["Value", "Error"]).T
print("\nOptimized Parameters for Black Hole Information Loss:")
print(df_optimal_params)
