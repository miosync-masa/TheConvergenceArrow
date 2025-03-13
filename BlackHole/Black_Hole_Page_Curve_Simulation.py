# Black Hole Page Curve Simulation

# ---- Explanation ----
# Purpose of this code:
# This script simulates the evolution of the Black Hole Page Curve under the influence of
# information convergence, quantified by λc. It also models the correlation between
# mutual information I(A;B) and the process of information recovery from black holes.
#
# Interpretation of Results:
# - The standard Page curve represents the entropy growth without external corrections.
# - The modified Page curve accounts for λc, which slows entropy growth as information convergence increases.
# - Mutual Information I(A;B) increases over time, indicating that higher λc accelerates information retrieval.
#
# Possible Adjustments:
# - Further refinement of λc dependence on entropy dynamics.
# - Investigating alternative models for mutual information in black hole evaporation.
# - Expanding the analysis to consider different scenarios of black hole information conservation.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulation parameters
time_steps = np.linspace(0, 1, 100)  # Normalized time (t/T_Page)
lambda_c_values = [0.0, 0.1, 0.3, 0.5]  # Different cases of convergence rate λc

# Standard Page curve model (without information recovery)
def standard_page_curve(t):
    return 2 * t / (1 + t)

# Modified Page curve model (with information convergence theory)
def modified_page_curve(t, lambda_c):
    return (1 / (1 + lambda_c)) * standard_page_curve(t)

# Plotting the Page curves
plt.figure(figsize=(8, 5))

# Plot modified Page curves for different λc values
for lambda_c in lambda_c_values:
    plt.plot(time_steps, modified_page_curve(time_steps, lambda_c), label=f"λc = {lambda_c}")

# Plot the standard Page curve
plt.plot(time_steps, standard_page_curve(time_steps), 'k--', label="Standard Page Curve")

# Labels and title
plt.xlabel("Normalized Time (t/T_Page)")
plt.ylabel("Entropy S")
plt.title("Black Hole Page Curve with Information Convergence")
plt.legend()
plt.grid()

# Show the plot
plt.show()

# Output numerical data
df_page_curve = pd.DataFrame({
    "Time (t/T_Page)": time_steps,
    "Standard Page Curve": standard_page_curve(time_steps),
})

# Add modified Page curve data for each λc
for lambda_c in lambda_c_values:
    df_page_curve[f"Modified Page Curve (λc={lambda_c})"] = modified_page_curve(time_steps, lambda_c)

# Display numerical data
print("\nSimulated Page Curve Data:")
print(df_page_curve.head(10))  # Display first 10 rows

# -------------------------------------------------------------
# Analysis of Mutual Information I(A;B) and Black Hole Information Recovery
# -------------------------------------------------------------

# Mutual information model (Assumption: I(A;B) increases as information is recovered)
def mutual_information_black_hole(t, lambda_c):
    return (1 - np.exp(-lambda_c * t)) * standard_page_curve(t)

# Plotting Mutual Information and Information Recovery
plt.figure(figsize=(8, 5))

for lambda_c in lambda_c_values:
    plt.plot(time_steps, mutual_information_black_hole(time_steps, lambda_c), label=f"λc = {lambda_c}")

plt.xlabel("Normalized Time (t/T_Page)")
plt.ylabel("Mutual Information I(A;B)")
plt.title("Mutual Information and Black Hole Information Recovery")
plt.legend()
plt.grid()

# Show the plot
plt.show()

# Output numerical data
df_mutual_info_bh = pd.DataFrame({
    "Time (t/T_Page)": time_steps,
})

# Add mutual information data for each λc
for lambda_c in lambda_c_values:
    df_mutual_info_bh[f"I(A;B) (λc={lambda_c})"] = mutual_information_black_hole(time_steps, lambda_c)

# Display numerical data
print("\nSimulated Mutual Information I(A;B) Data:")
print(df_mutual_info_bh.head(10))  # Display first 10 rows
