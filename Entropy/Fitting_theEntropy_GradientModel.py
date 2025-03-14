# Fitting_theEntropy_GradientModel with Information Convergence Rate

# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Existing data (λc vs. dS/dλc)
lambda_c_fit_values = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])  # Information convergence rate (λc)
dS_dlambda_c_values = np.array([-0.527718, -0.492488, -0.435971, -0.362055, -0.239899, -0.126113])  # Observed entropy gradient

# Define the theoretical entropy gradient model
def theoretical_entropy_grad(lambda_c, S0, alpha):
    """
    Computes the theoretical entropy gradient as a function of the information convergence rate (λc).

    Parameters:
    - lambda_c: Information convergence rate (array)
    - S0: Initial entropy parameter
    - alpha: Decay parameter controlling the rate of entropy change

    Returns:
    - Modeled entropy gradient (dS/dλc)
    """
    return -S0 / ((1 + lambda_c) ** 2) * np.exp(-alpha * lambda_c)

# Initial guess for fitting parameters
initial_guess = [max(abs(dS_dlambda_c_values)), 0.1]  # S0, α (scaling and decay parameters)

# Perform curve fitting
params_theory, _ = curve_fit(theoretical_entropy_grad, lambda_c_fit_values, dS_dlambda_c_values, p0=initial_guess)

# Extract fitted parameters
S0_fit, alpha_fit = params_theory

# Print the fitted parameters
print(f"\nFitted Parameters:")
print(f"S₀ = {S0_fit:.5f}")  # Fitted initial entropy parameter
print(f"α  = {alpha_fit:.5f}")  # Fitted decay parameter

# Generate a wider range of λc values for theoretical model plotting
lambda_c_range = np.linspace(0.0001, 0.1, 100)
grad_fit = theoretical_entropy_grad(lambda_c_range, S0_fit, alpha_fit)

# 📊 Plot the observed data vs. theoretical fit
plt.figure(figsize=(8, 5))

# Scatter plot for observed data
plt.scatter(lambda_c_fit_values, dS_dlambda_c_values, color='red', label="Observed Data")

# Plot for fitted model
plt.plot(lambda_c_range, grad_fit, 'b-', label="Fitted Model")

# Configure plot labels and title
plt.xlabel("Information Convergence Rate (λc)")
plt.ylabel("Entropy Gradient (dS/dλc)")
plt.title("Comparison of Theoretical Model with Observed Data")

# Add legend and grid for better visualization
plt.legend()
plt.grid()

# Show the plot
plt.show()

# 📜 Create a DataFrame to compare observed vs. fitted values
df_fit_results = pd.DataFrame({
    "λc": lambda_c_fit_values,
    "Observed dS/dλc": dS_dlambda_c_values,
    "Fitted dS/dλc": theoretical_entropy_grad(lambda_c_fit_values, S0_fit, alpha_fit)
})

# Display the comparison of observed and fitted values
print("\nComparison of Observed and Fitted dS/dλc Values:")
print(df_fit_results)
