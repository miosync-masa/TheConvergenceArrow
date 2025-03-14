# TimeScaling_with_Information Convergence and Transaction Density

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the range of the information convergence rate (λc)
lambda_c_values = np.linspace(0, 5, 100)  # λc values from 0 to 5
rho_T_values = [0.1, 0.575, 1.05, 1.525, 2.0]  # Different transaction densities (ρT)

# Define the new time scaling function based on λc and ρT
def new_time_definition(lambda_c, rho_T, T0, beta):
    """
    Computes the modified time scale as a function of the 
    information convergence rate (λc) and transaction density (ρT).
    
    Parameters:
    - lambda_c: Information convergence rate (array)
    - rho_T: Transaction density (scalar)
    - T0: Baseline time reference
    - beta: Scaling factor controlling the impact of λc

    Returns:
    - Computed time values t(λc, ρT)
    """
    return T0 * np.exp(beta * lambda_c) / rho_T  # Exponential scaling with λc, compressed by ρT

# Set parameter values
T0 = 1.0  # Baseline time reference (dimensionless)
beta = 1.1  # Strength of the information convergence effect

# Initialize the figure for plotting
plt.figure(figsize=(8, 5))

# Compute and plot the modified time scale for different ρT values
for rho_T in rho_T_values:
    t_values = new_time_definition(lambda_c_values, rho_T, T0, beta)
    plt.plot(lambda_c_values, t_values, label=f"ρ_T = {rho_T}")

# Configure the plot labels and title
plt.xlabel("Information Convergence Rate (λc)")
plt.ylabel("Time t(λc, ρ_T)")
plt.title("Time Scaling with Information Convergence and Transaction Density")

# Use a logarithmic scale for better visualization of exponential effects
plt.yscale("log")

# Add legend and grid for better readability
plt.legend()
plt.grid()

# Display the plot
plt.show()

# Create a DataFrame to store simulated data
df_time = pd.DataFrame({
    "Lambda_c": lambda_c_values
})

# Compute and store time values for different ρT values
for rho_T in rho_T_values:
    df_time[f"Time (ρ_T={rho_T})"] = new_time_definition(lambda_c_values, rho_T, T0, beta)

# Display the first 10 rows of the dataset
print("\nSimulated Time Function Data:")
print(df_time.head(10))  # Show the first 10 rows
