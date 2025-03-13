# BlackHoleInformationLossSimulation

# ---- Explanation ----
# Purpose of this code:
# This script simulates information loss in black holes, incorporating entanglement effects and Hawking temperature.
# The model assumes that information loss is influenced by black hole mass M, entanglement strength E, and a decay parameter α.
#
# Key Concept:
# - Hawking temperature T_H decreases with increasing black hole mass M.
# - Information loss is exponentially suppressed by entanglement (E) and the Hawking temperature correction factor.
# - The model suggests that stronger entanglement slows information loss, potentially stabilizing black hole information retention.
#
# Interpretation of Results:
# - Lower M leads to higher Hawking temperature, increasing information loss.
# - Higher entanglement strength E reduces information loss exponentially.
# - The contour plot shows regions of high and low information loss based on the parameters.
#
# Possible Adjustments:
# - Modifying α to explore different decay rates.
# - Extending the model to incorporate transaction density effects.
# - Comparing results with observational data on black hole information retention.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameter settings
M_values = np.linspace(1, 10, 100)  # Range of black hole masses
E_values = np.linspace(0, 1, 100)  # Range of entanglement strength

# Hawking temperature calculation
def hawking_temperature(M):
    return 1 / (8 * np.pi * M)

# Modified information loss model
def information_loss(M, E, alpha):
    T_H = hawking_temperature(M)
    return -1 / M**2 * np.exp(-alpha / T_H) * np.exp(-alpha * E)

# Compute information loss across parameters
loss_data = np.array([[information_loss(M, E, 0.5) for E in E_values] for M in M_values])

# Avoid log issues by replacing zeros with small values
loss_data[loss_data > -1e-10] = -1e-10

# Create a DataFrame for results
df_info_loss = pd.DataFrame(loss_data, index=M_values, columns=E_values)

# Plot the results with log scale
plt.figure(figsize=(8, 5))
contour = plt.contourf(E_values, M_values, np.log10(-loss_data), levels=20, cmap="viridis")
plt.colorbar(label="Log10(Information Loss Rate)")
plt.xlabel("Entanglement Strength E")
plt.ylabel("Black Hole Mass M")
plt.title("Information Loss vs. Black Hole Mass and Entanglement Strength (Log Scale)")
plt.show()

# Display numerical data (first 10 rows and columns)
print("\nSimulated Information Loss Data:")
print(df_info_loss.iloc[:10, :10])  # Display first 10x10 block of data
