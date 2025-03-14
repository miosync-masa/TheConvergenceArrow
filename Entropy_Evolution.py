#Entropy Evolution and Transaction-Based Scaling of Convergence Rate 

# Required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.optimize import curve_fit

# Enable inline plotting for Google Colab
%matplotlib inline

# Existing Quantum Zeno Effect (QZE) data (Observation Frequency vs. Survival Probability)
obs_freq = np.array([1, 5, 10, 20, 50])  # Observation frequency in Hz
survival_prob = np.array([0.95, 0.85, 0.75, 0.60, 0.40])  # Survival probability

# Compute Shannon entropy values based on survival probability
entropy_values = -survival_prob * np.log(survival_prob)

# Entropy growth model based on the Convergence Arrow framework
def entropy_lambda_model(f, S0, lambda_c, alpha):
    """
    Computes entropy evolution as a function of observation frequency (f)
    based on the information convergence rate (λc).

    Parameters:
    - f: Observation frequency (Hz)
    - S0: Initial entropy
    - lambda_c: Information convergence rate
    - alpha: Scaling factor controlling the effect of observation frequency

    Returns:
    - Modeled entropy values
    """
    return S0 / (1 + lambda_c * np.log(1 + alpha * f))

# Initial parameter guesses and bounds for curve fitting
initial_guess_lambda = [max(entropy_values), 0.005, 10]  # S0, λc, α
bounds_lambda = ([0, 0.001, 1], [np.inf, 1.0, 200])  # Constraining λc between 0.001 and 1.0

# Perform curve fitting
params_lambda, _ = curve_fit(
    entropy_lambda_model, 
    obs_freq, 
    entropy_values, 
    p0=initial_guess_lambda, 
    bounds=bounds_lambda,
    maxfev=100000  # Increase max iterations for better convergence
)

# Extract fitted parameters
S0_lambda, lambda_c_fit, alpha_fit = params_lambda

# Assume energy values proportional to entropy changes (dimensionless)
E_values = entropy_values  

# Compute transaction density ρT using fitted λc
C_value = 1  # Scaling factor (dimensionless)
rho_T_values = E_values / (C_value * lambda_c_fit)

# Compute change in ρT (ΔρT)
d_rho_T = np.diff(rho_T_values)

# Compute change in λc (Δλc) assuming a dependency on ρT
alpha_coefficient = 0.1  # Scaling factor for λc evolution
d_lambda_c = -alpha_coefficient * lambda_c_fit * d_rho_T

# Compute the rate of change dλc/dρT
lambda_progress_per_rho = d_lambda_c / d_rho_T

# Construct DataFrame for analysis
df_progress_analysis = pd.DataFrame({
    "Observation Frequency (Hz)": obs_freq[:-1],  # Adjust size to match ΔρT
    "ΔρT": d_rho_T,
    "Δλc": d_lambda_c,
    "dλc/dρT": lambda_progress_per_rho
})

# 🔥 Scale Δλc for better visualization
scale_factor = 10**3  # Convert scale to 10^3
df_progress_analysis["Δλc_scaled"] = df_progress_analysis["Δλc"] * scale_factor

# 🔥 Exponential model definition (including an offset term c)
def exp_model(x, a, b, c):
    """
    Defines an exponential decay model with an offset.

    Parameters:
    - x: Input variable
    - a, b, c: Fit parameters

    Returns:
    - Exponential function output
    """
    return a * np.exp(-b * x) + c  # Adding c to account for asymptotic behavior

# 🔥 Perform curve fitting with increased iterations for better convergence
try:
    params, _ = curve_fit(
        exp_model, 
        df_progress_analysis["ΔρT"], 
        df_progress_analysis["Δλc"], 
        p0=[-1, 0.01, 0],  # Initial guesses
        maxfev=200000  # Increased max iterations
    )
except RuntimeError as e:
    print(f"⚠️ Curve fitting failed: {e}")
    params = [-1, 0.01, 0]  # Default values if fitting fails

# Apply fitted model to the data
df_progress_analysis["Fitted_Δλc"] = exp_model(df_progress_analysis["ΔρT"], *params)

# 📊 **Plot the relationship between ΔρT and λc progress using log-log scale**
plt.figure(figsize=(8, 5))

# Use absolute values to enable log scaling
plt.scatter(df_progress_analysis["ΔρT"], np.abs(df_progress_analysis["Δλc"]), color='green', label="|λc Progress per ρT Change|")
plt.plot(df_progress_analysis["ΔρT"], np.abs(df_progress_analysis["Fitted_Δλc"]), 'r--', label="Exponential Fit")

# Use log-log scale to highlight scaling behavior
plt.xscale("log")
plt.yscale("log")
plt.xlabel("ΔρT (Change in Transaction Density)")
plt.ylabel("|Δλc| (Change in Progress Bar)")
plt.title("Log-Log Scale: Relationship between ρT Change and λc Progress")
plt.legend()
plt.grid()
plt.show()

# 📜 **Display the DataFrame with the computed values**
print("🔥 Convergence Progress Analysis DataFrame 🔥")
display(df_progress_analysis)
