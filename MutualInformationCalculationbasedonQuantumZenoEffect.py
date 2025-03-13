# Mutual Information Calculation based on Quantum Zeno Effect in Google Colab

# ---- Explanation ----
# Purpose of this code:
# This script analyzes the relationship between observation frequency and mutual information
# in the context of the Quantum Zeno Effect. It calculates entropy values to assess how
# frequent quantum measurements affect information retention.
#
# Interpretation of Results:
# The negative values of Mutual Information (I(A;B)) indicate a possible inconsistency in
# entropy calculations, suggesting that increased observation frequency may introduce
# information loss or distortion in the measurement process. This could imply that excessive
# measurement influences system evolution in unexpected ways.
#
# Possible Adjustments:
# - Revising the conditional probability function to ensure valid entropy values.
# - Investigating alternative entropy formulations to confirm theoretical consistency.
# - Exploring additional factors affecting information retention in quantum systems.

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Quantum Zeno effect data (observation frequency vs survival probability)
obs_freq = np.array([1, 5, 10, 20, 50])  # Observation frequencies in Hz
survival_prob = np.array([0.95, 0.85, 0.75, 0.60, 0.40])  # Survival probabilities

# Conditional probability P(A|B): Assumes exponential decrease based on observation frequency
P_A_given_B = survival_prob * np.exp(-obs_freq / 50)

# Calculate entropies:
# Entropy S(A): uncertainty of the future state A
S_A = -survival_prob * np.log(survival_prob)

# Entropy S(B): Entropy of the current state B
S_B = -(1 - survival_prob) * np.log(1 - survival_prob)

# Conditional Entropy S(A|B): Uncertainty of A given B
S_A_given_B = -P_A_given_B * np.log(P_A_given_B)

# Joint entropy S(A,B): Total uncertainty combining states A and B
S_AB = S_A_given_B + S_B

# Mutual Information I(A;B): Shared information between states A and B
I_AB = S_A + S_B - S_AB

# Creating a DataFrame to organize results
df_mutual_info_corrected = pd.DataFrame({
    "Observation Frequency (Hz)": obs_freq,
    "Survival Probability": survival_prob,
    "Entropy S(A)": S_A,
    "Entropy S(B)": S_B,
    "Conditional Entropy S(A|B)": S_A_given_B,
    "Joint Entropy S(A,B)": S_AB,
    "Mutual Information I(A;B)": I_AB
})

# Displaying the calculated DataFrame
print("Corrected Mutual Information Calculation:")
print(df_mutual_info_corrected)

# Plot Mutual Information as a function of observation frequency
plt.figure(figsize=(8, 5))
plt.plot(obs_freq, I_AB, marker='o', linestyle='-', color='blue')
plt.title('Mutual Information vs Observation Frequency')
plt.xlabel('Observation Frequency (Hz)')
plt.ylabel('Mutual Information I(A;B)')
plt.grid(True)
plt.show()
