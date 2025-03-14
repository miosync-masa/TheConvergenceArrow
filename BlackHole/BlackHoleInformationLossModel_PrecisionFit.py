# BlackHoleInformationLossModel_PrecisionFit

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Black Hole Information Loss Model ---
def S_BH(t, S0, lambda_c, rho_T):
    """
    Computes black hole entropy evolution over time.
    
    Parameters:
    - t: Time variable
    - S0: Maximum entropy (final entropy state)
    - lambda_c: Convergence rate of information loss
    - rho_T: Transaction density (rate of information transactions)
    
    Returns:
    - S_BH: Black hole entropy as a function of time
    """
    return S0 * (1 - np.exp(-lambda_c * rho_T * t))

# --- Generate Simulated Data ---
t_data = np.linspace(0, 10, 50)  # Time range
S0_true = 1.0  # Theoretical maximum entropy
lambda_c_true = 0.15  # Assumed true value for λc
rho_T_true = 0.8  # Assumed true transaction density

# Generate simulated entropy data with added noise
S_data = S_BH(t_data, S0_true, lambda_c_true, rho_T_true) + np.random.normal(0, 0.02, len(t_data))

# --- Perform Curve Fitting to Estimate Parameters ---
popt, pcov = curve_fit(S_BH, t_data, S_data, p0=[1.0, 0.1, 1.0])

# Extract best-fit parameters
S0_fit, lambda_c_fit, rho_T_fit = popt

# --- Generate Data for Fitted Model ---
t_fit = np.linspace(0, 10, 100)  # Higher resolution time steps
S_fit = S_BH(t_fit, S0_fit, lambda_c_fit, rho_T_fit)

# --- Plot the Results ---
plt.figure(figsize=(8, 5))
plt.scatter(t_data, S_data, color='red', label='Simulated Data')  # Observed entropy data
plt.plot(t_fit, S_fit, 'b-', 
         label=f'Fitted Model: $S_{{BH}}(t) = S_0 (1 - e^{{-\\lambda_c \\rho_T t}})$\n'
               f'$\\lambda_c$ = {lambda_c_fit:.4f}, $\\rho_T$ = {rho_T_fit:.4f}')
plt.xlabel('Time (t)')
plt.ylabel('Black Hole Entropy $S_{BH}$')
plt.title('Black Hole Information Loss Model - Precision Fit')
plt.legend()
plt.grid()
plt.show()

# --- Display Best-Fit Parameters ---
S0_fit, lambda_c_fit, rho_T_fit
