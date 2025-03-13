# BlackHole_EvaporationTimeScaleSimulation_with Information Convergence

# ---- Explanation ----
# Purpose of this code:
# This script simulates black hole evaporation and investigates how information convergence (λc)
# alters the evaporation rate. The model is compared to the standard Hawking evaporation model.
# A curve-fitting approach is applied to estimate λc from observed data.
#
# Key Concept:
# - The standard Hawking model follows M(T) ~ 1/(1 + T), implying gradual mass loss.
# - The modified model introduces λc, affecting the rate of mass loss by slowing evaporation.
# - The fitting process attempts to extract the best estimate of λc from a given dataset.
#
# Interpretation of Results:
# - If λc > 0, evaporation is slower, meaning information convergence prolongs black hole existence.
# - The fitting results provide an estimated λc, useful for comparing theoretical models to observations.
# - If λc trends significantly upward, it suggests black hole information retention effects.
#
# Possible Adjustments:
# - Expanding the model to incorporate entropy and transaction density.
# - Testing with different initial black hole masses and entropy constraints.
# - Incorporating observational constraints to validate λc estimations.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Initial black hole mass (in solar masses)
M_initial = 5  # Example for a small black hole
time_steps = np.linspace(0, 10, 500)  # Normalized time (T/T_Page)
lambda_c_values = [0.0, 0.1, 0.3, 0.5, 1.0]  # Different cases of convergence rate λc

# Standard Hawking Radiation Evaporation Time Scale
def standard_evaporation_time(M0):
    return M0**3  # Hawking evaporation time scale ~ M^3

# Modified Evaporation Time Scale with Information Convergence
def modified_evaporation_time(M0, lambda_c):
    return (1 + lambda_c) * M0**3  # Considering the effect of information convergence

# New Evaporation Model with Convergence Arrow Theory (for fitting)
def evaporation_model(t, M0, lambda_c):
    return M0 / (1 + lambda_c * t)  # Model incorporating information convergence

# Plot evaporation time scale
plt.figure(figsize=(8, 5))

# Plot modified Hawking radiation curves for different λc values
for lambda_c in lambda_c_values:
    lifetime = modified_evaporation_time(M_initial, lambda_c)
    plt.plot(time_steps, evaporation_model(time_steps, M_initial, lambda_c), label=f"λc = {lambda_c}")

# Plot standard Hawking radiation curve
plt.plot(time_steps, M_initial / (1 + time_steps), 'k--', label="Standard Hawking Radiation")

# Labels and title
plt.xlabel("Normalized Time (T/T_Page)")
plt.ylabel("Black Hole Mass M(T)")
plt.title("Black Hole Evaporation Time Scale with Information Convergence")
plt.legend()
plt.grid()

# Show the plot
plt.show()

# Output numerical data
df_evaporation_time = pd.DataFrame({
    "Time (T/T_Page)": time_steps,
    "Standard Evaporation": M_initial / (1 + time_steps)
})

# Add modified evaporation time data for each λc
for lambda_c in lambda_c_values:
    df_evaporation_time[f"Modified Evaporation (λc={lambda_c})"] = evaporation_model(time_steps, M_initial, lambda_c)

# Display numerical data
print("\nSimulated Black Hole Evaporation Time Scale Data:")
print(df_evaporation_time.head(10))  # Display first 10 rows

# -------------------------------------------------------------
# Fitting Execution
# -------------------------------------------------------------

# Generate observed data (simulation-based example)
observed_time = time_steps
observed_mass = M_initial / (1 + 0.2 * time_steps)  # Assume λc = 0.2 as observed data

# Perform curve fitting
params, covariance = curve_fit(evaporation_model, observed_time, observed_mass, p0=[M_initial, 0.1])

# Extract fitted parameters
M0_fit, lambda_c_fit = params
print(f"\nFitted Parameters:")
print(f"Estimated M0 = {M0_fit:.5f}")
print(f"Estimated λc = {lambda_c_fit:.5f}")

# Generate fitted model for visualization
fitted_mass = evaporation_model(time_steps, M0_fit, lambda_c_fit)

# Plot observed data and fitted model
plt.figure(figsize=(8, 5))
plt.scatter(observed_time, observed_mass, color='red', label="Observed Data")
plt.plot(time_steps, fitted_mass, 'b-', label="Fitted Model")
plt.xlabel("Normalized Time (T/T_Page)")
plt.ylabel("Black Hole Mass M(T)")
plt.title("Fitting of Black Hole Evaporation Model")
plt.legend()
plt.grid()
plt.show()

