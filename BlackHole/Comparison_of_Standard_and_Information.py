# Comparison_of_Standard_and_Information-Convergent Black Hole Evaporation Models

# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Initial mass of the black hole (in solar mass units)
M_initial = 5  # Example: small black hole

# Normalized time steps (T/T_Page)
time_steps = np.linspace(0, 10, 500)

# Different values of the convergence rate λc
lambda_c_values = [0.0, 0.1, 0.3, 0.5, 1.0]

# Standard Hawking evaporation time scale (M^3 dependence)
def standard_evaporation_time(M0):
    """
    Computes the standard Hawking radiation evaporation time scale.

    Parameters:
    - M0: Initial black hole mass

    Returns:
    - Evaporation time scale proportional to M^3
    """
    return M0**3

# Modified evaporation time scale incorporating the convergence rate λc
def modified_evaporation_time(M0, lambda_c):
    """
    Computes the modified black hole evaporation time scale incorporating
    information convergence effects.

    Parameters:
    - M0: Initial black hole mass
    - lambda_c: Information convergence rate

    Returns:
    - Adjusted evaporation time scale
    """
    return (1 + lambda_c) * M0**3

# Evaporation model incorporating the Convergence Arrow Theory
def evaporation_model(t, M0, lambda_c):
    """
    Models black hole mass evolution under the influence of information convergence.

    Parameters:
    - t: Normalized time (T/T_Page)
    - M0: Initial black hole mass
    - lambda_c: Information convergence rate

    Returns:
    - Modeled black hole mass M(T)
    """
    return M0 / (1 + lambda_c * t)

# 📊 Plot the black hole evaporation time scale
plt.figure(figsize=(8, 5))

# Plot evaporation curves for different values of λc
for lambda_c in lambda_c_values:
    lifetime = modified_evaporation_time(M_initial, lambda_c)
    plt.plot(time_steps, evaporation_model(time_steps, M_initial, lambda_c), label=f"λc = {lambda_c}")

# Plot the standard Hawking radiation curve for comparison
plt.plot(time_steps, M_initial / (1 + time_steps), 'k--', label="Standard Hawking Radiation")

# Configure plot labels and title
plt.xlabel("Normalized Time (T/T_Page)")
plt.ylabel("Black Hole Mass M(T)")
plt.title("Black Hole Evaporation Time Scale with Information Convergence")

# Add legend and grid for better visualization
plt.legend()
plt.grid()

# Show the plot
plt.show()

# 📜 Create a DataFrame to store simulated data
df_evaporation_time = pd.DataFrame({
    "Time (T/T_Page)": time_steps,
    "Standard Evaporation": M_initial / (1 + time_steps)
})

# Compute and store modified evaporation data for each λc
for lambda_c in lambda_c_values:
    df_evaporation_time[f"Modified Evaporation (λc={lambda_c})"] = evaporation_model(time_steps, M_initial, lambda_c)

# Display the first 10 rows of the dataset
print("\nSimulated Black Hole Evaporation Time Scale Data:")
print(df_evaporation_time.head(10))

# -------------------------------------------------------------
# Performing Curve Fitting to Estimate λc from Observed Data
# -------------------------------------------------------------

# Generate simulated observed data (assuming λc = 0.2 for testing)
observed_time = time_steps
observed_mass = M_initial / (1 + 0.2 * time_steps)

# Perform curve fitting to estimate λc
params, covariance = curve_fit(evaporation_model, observed_time, observed_mass, p0=[M_initial, 0.1])

# Extract fitted parameters
M0_fit, lambda_c_fit = params

# Print fitted parameters
print(f"\nFitted Parameters:")
print(f"Estimated M0 = {M0_fit:.5f}")
print(f"Estimated λc = {lambda_c_fit:.5f}")

# 📊 Plot the observed vs. fitted evaporation model
fitted_mass = evaporation_model(time_steps, M0_fit, lambda_c_fit)

plt.figure(figsize=(8, 5))

# Scatter plot for observed data
plt.scatter(observed_time, observed_mass, color='red', label="Observed Data")

# Plot for fitted model
plt.plot(time_steps, fitted_mass, 'b-', label="Fitted Model")

# Configure plot labels and title
plt.xlabel("Normalized Time (T/T_Page)")
plt.ylabel("Black Hole Mass M(T)")
plt.title("Fitting of Black Hole Evaporation Model")

# Add legend and grid for better visualization
plt.legend()
plt.grid()

# Show the plot
plt.show()
