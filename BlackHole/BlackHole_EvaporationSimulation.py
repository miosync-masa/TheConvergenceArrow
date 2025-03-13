# BlackHole_EvaporationSimulation

# ---- Explanation ----
# Purpose of this code:
# This script simulates black hole evaporation with and without information convergence effects.
# The standard Hawking radiation model assumes mass loss follows M(T) = M0 / (1 + T).
# The modified model incorporates λc, which alters the evaporation rate based on information convergence.
#
# Key Concept:
# - The higher the λc, the slower the black hole evaporation due to stronger information retention.
# - Information convergence (λc) acts as a delay factor, modifying the rate of mass loss.
# - The modified model suggests that black hole evaporation can be influenced by information recovery.
#
# Interpretation of Results:
# - In the standard model, black hole mass decreases steadily over time.
# - As λc increases, the evaporation slows, indicating that information retention counteracts mass loss.
# - This supports the hypothesis that black hole evaporation is not purely thermodynamic but can be altered by information effects.
#
# Possible Adjustments:
# - Further exploration of λc dynamics and how it interacts with black hole entropy.
# - Incorporating observational constraints to test model accuracy.
# - Investigating how transaction density influences the evaporation rate in extreme conditions.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Black Hole Evaporation Simulation Parameters
time_steps = np.linspace(0, 10, 100)  # Normalized time (T/T_Page)
lambda_c_values = [0.0, 0.1, 0.3, 0.5]  # Different cases of convergence rate λc

# Standard Hawking Radiation Model (without information convergence)
def standard_evaporation_rate(t, M0=1):
    return M0 / (1 + t)

# Modified Hawking Radiation Model (with information convergence theory)
def modified_evaporation_rate(t, lambda_c, M0=1):
    return M0 / (1 + (1 + lambda_c) * t)

# Plotting the evaporation curves
plt.figure(figsize=(8, 5))

# Plot modified Hawking radiation curves for different λc values
for lambda_c in lambda_c_values:
    plt.plot(time_steps, modified_evaporation_rate(time_steps, lambda_c), label=f"λc = {lambda_c}")

# Plot the standard Hawking radiation curve
plt.plot(time_steps, standard_evaporation_rate(time_steps), 'k--', label="Standard Hawking Radiation")

# Labels and title
plt.xlabel("Normalized Time (T/T_Page)")
plt.ylabel("Black Hole Mass M(T)")
plt.title("Black Hole Evaporation with Information Convergence")
plt.legend()
plt.grid()

# Show the plot
plt.show()

# Output numerical data
df_evaporation = pd.DataFrame({
    "Time (T/T_Page)": time_steps,
    "Standard Evaporation": standard_evaporation_rate(time_steps),
})

# Add modified evaporation data for each λc
for lambda_c in lambda_c_values:
    df_evaporation[f"Modified Evaporation (λc={lambda_c})"] = modified_evaporation_rate(time_steps, lambda_c)

# Display numerical data
print("\nSimulated Black Hole Evaporation Data:")
print(df_evaporation.head(10))  # Display first 10 rows
