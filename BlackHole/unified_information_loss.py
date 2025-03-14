# Unified Information Loss Model

# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the range of information convergence rate (λc)
lambda_c_values = np.linspace(0, 1, 100)  # λc values from 0 to 1

# Unified Information Loss Model (for Black Hole & Entanglement Collapse)
def unified_information_loss(lambda_c, S0, alpha):
    """
    Computes the normalized information loss based on the convergence rate (λc).

    Parameters:
    - lambda_c: Information convergence rate (array)
    - S0: Initial entropy value (normalization factor)
    - alpha: Decay parameter controlling information loss rate

    Returns:
    - Normalized information loss
    """
    return -S0 / ((1 + lambda_c) ** 2) * np.exp(-alpha * lambda_c)

# Set initial entropy values for different scenarios
S0_blackhole = 1.0  # Baseline entropy for black hole information loss
S0_entanglement = 1.0  # Baseline entropy for entanglement collapse

# Set decay parameters
alpha_blackhole = 0.5  # Decay parameter for black hole information loss
alpha_entanglement = 0.3  # Decay parameter for entanglement collapse

# Compute information loss for black hole and entanglement collapse
S_blackhole = unified_information_loss(lambda_c_values, S0_blackhole, alpha_blackhole)
S_entanglement = unified_information_loss(lambda_c_values, S0_entanglement, alpha_entanglement)

# 📊 Plot the information loss curves
plt.figure(figsize=(8, 5))

# Black Hole Information Loss
plt.plot(lambda_c_values, S_blackhole, label="Black Hole Information Loss", color='blue')

# Entanglement Collapse
plt.plot(lambda_c_values, S_entanglement, label="Entanglement Collapse", color='red')

# Configure plot labels and title
plt.xlabel("Information Convergence Rate (λc)")
plt.ylabel("Normalized Information Loss")
plt.title("Unified Model: Black Hole Information vs. Entanglement Collapse")

# Add legend and grid for better visualization
plt.legend()
plt.grid()

# Show the plot
plt.show()

# 📜 Create a DataFrame to store the computed values
df_unified = pd.DataFrame({
    "Lambda_c": lambda_c_values,
    "Black Hole Information Loss": S_blackhole,
    "Entanglement Collapse": S_entanglement
})

# Display the first 10 rows of the simulated data
print("\nSimulated Unified Information Loss Data:")
print(df_unified.head(10))  # Show only the first 10 rows
