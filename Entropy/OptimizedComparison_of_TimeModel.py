# OptimizedComparison_of_TimeModel and Convergence Arrow Model for Entropy Growth

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given QZE data (Observation Frequency vs. Survival Probability)
obs_freq = np.array([1, 5, 10, 20, 50])  # Measurement frequencies in Hz
survival_prob = np.array([0.95, 0.85, 0.75, 0.60, 0.40])  # Survival probability at each frequency

# Compute Shannon entropy based on survival probability
entropy_values = -survival_prob * np.log(survival_prob)

# --- Traditional entropy growth model based on time decay ---
def entropy_time_model(t, S0, tau):
    """
    Traditional time-dependent entropy model.
    
    Parameters:
    t (float): Observation frequency (Hz)
    S0 (float): Initial entropy value
    tau (float): Decay constant for entropy reduction
    
    Returns:
    float: Modeled entropy value
    """
    return S0 * np.exp(-t / tau)

# --- Convergence Arrow-based entropy model ---
def entropy_lambda_model(f, S0, lambda_c, alpha):
    """
    Entropy growth model based on Convergence Arrow theory.
    
    Parameters:
    f (float): Observation frequency (Hz)
    S0 (float): Initial entropy value
    lambda_c (float): Information convergence rate
    alpha (float): Scaling factor for transaction density impact
    
    Returns:
    float: Modeled entropy value
    """
    return S0 / (1 + lambda_c * np.log(1 + alpha * f))

# Initial parameter estimates with constraints on alpha and lambda_c
initial_guess_lambda = [max(entropy_values), 0.01, 10]  # Initial estimates for S0, lambda_c, and alpha
bounds_lambda = ([0, 0.001, 1], [np.inf, 1.0, 10000])  # Restrict alpha between 1 and 10000

# Fit the Lambda-dependent entropy model with constraints
params_lambda, _ = curve_fit(
    entropy_lambda_model,
    obs_freq,
    entropy_values,
    p0=initial_guess_lambda,
    bounds=bounds_lambda,
    maxfev=20000  # Increase the maximum number of function evaluations
)

# Fit the traditional Time-dependent entropy model
params_time, _ = curve_fit(entropy_time_model, obs_freq, entropy_values, p0=[max(entropy_values), 10])

# Extract fitted parameters for both models
S0_lambda, lambda_c_fit, alpha_fit = params_lambda  # Convergence Arrow model parameters
S0_time, tau_time = params_time  # Time-dependent model parameters

# Compute RMSE (Root Mean Square Error) for model comparison
rmse_time = np.sqrt(np.mean((entropy_values - entropy_time_model(obs_freq, *params_time)) ** 2))
rmse_lambda = np.sqrt(np.mean((entropy_values - entropy_lambda_model(obs_freq, *params_lambda)) ** 2))

# Generate data for visualization
freq_range = np.linspace(min(obs_freq), max(obs_freq), 500)  # Interpolated frequency range
entropy_fit_time = entropy_time_model(freq_range, *params_time)  # Time model prediction
entropy_fit_lambda = entropy_lambda_model(freq_range, *params_lambda)  # Lambda model prediction

# --- Visualization of observed and fitted models ---
plt.figure(figsize=(8, 5))
plt.scatter(obs_freq, entropy_values, color='red', label="Observed Data")  # Plot observed entropy values
plt.plot(freq_range, entropy_fit_time, 'b--', label=f"Time Model (RMSE={rmse_time:.4f})")  # Time model
plt.plot(freq_range, entropy_fit_lambda, 'g-', label=f"Lambda Model (RMSE={rmse_lambda:.4f})")  # Lambda model
plt.xlabel("Observation Frequency (Hz)")
plt.ylabel("Entropy (Shannon Approximation)")
plt.title("Entropy Growth: Time Model vs Lambda Model (Optimized)")
plt.legend()
plt.grid()
plt.show()

# --- Create a DataFrame for model comparison ---
df_error_analysis = pd.DataFrame({
    "Observation Frequency (Hz)": obs_freq,
    "Observed Entropy": entropy_values,
    "Time Model Entropy": entropy_time_model(obs_freq, *params_time),
    "Lambda Model Entropy": entropy_lambda_model(obs_freq, *params_lambda)
})

# Display the DataFrame in Colab (Print instead of ace_tools)
print("\n🔥 Optimized Entropy Model Comparison 🔥")
print(df_error_analysis.head(10))  # Display first 10 rows

# Return optimized parameters for further analysis
(S0_lambda, lambda_c_fit, alpha_fit, rmse_lambda)
