# BlackHole_EvaporationTime_Simulation_with_Information_Convergence.py

# ---- Explanation ----
# Purpose of this code:
# This script analyzes the relationship between the information convergence rate (λc) and
# black hole evaporation time. It assumes that as λc increases, evaporation slows down due to
# enhanced information retention.
#
# Interpretation of Results:
# - The fitted model follows an exponential function, suggesting that information convergence
#   exponentially extends the black hole lifetime.
# - A higher λc results in significantly longer evaporation times, implying a stronger influence
#   of information on black hole thermodynamics.
#
# Possible Adjustments:
# - Testing alternative functional forms for evaporation time dependence on λc.
# - Incorporating additional observational constraints to refine model accuracy.
# - Expanding the parameter space to include variations in black hole mass and energy dissipation rates.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Provided data (λc vs. Black Hole Evaporation Time)
lambda_c_values = np.array([0.0, 0.1, 0.3, 0.5, 1.0])
T_evap_values = np.array([5.0, 5.5, 7.0, 9.0, 15.0])  # Example: Evaporation time under Convergence Arrow Model

# Exponential Model for Evaporation Time
def evap_time_model(lambda_c, T0, alpha):
    return T0 * np.exp(alpha * lambda_c)

# Curve fitting execution
initial_guess = [5.0, 1.0]  # Initial estimates for (T0, α)
params, covariance = curve_fit(evap_time_model, lambda_c_values, T_evap_values, p0=initial_guess)

# Extract fitted parameters
T0_fit, alpha_fit = params
print(f"\nFitted Parameters:")
print(f"Estimated T0 = {T0_fit:.5f}")
print(f"Estimated α = {alpha_fit:.5f}")

# Generate fitted model for visualization
lambda_c_range = np.linspace(0, 1, 100)
T_evap_fit = evap_time_model(lambda_c_range, T0_fit, alpha_fit)

# Plot observed data and fitted model
plt.figure(figsize=(8, 5))
plt.scatter(lambda_c_values, T_evap_values, color='red', label="Observed Data")
plt.plot(lambda_c_range, T_evap_fit, 'b-', label="Fitted Model")
plt.xlabel("Information Convergence Rate (λc)")
plt.ylabel("Black Hole Evaporation Time T_evap")
plt.title("Fitting of Black Hole Evaporation Model with Information Convergence")
plt.legend()
plt.grid()
plt.show()
