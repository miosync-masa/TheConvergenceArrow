import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- DefinetheEnergyScale Domain ---
# Energy scale spans from 10^-3 eV to 10^3 eV, covering a broad range of quantum interactions.
energy_scale = np.logspace(-3, 3, 100)  

# --- Hypothetical Model for Convergence Rate λ_c as a Function of Energy Scale ---
def lambda_c_model(E, a, b):
    """
    Model for the convergence rate λ_c as a function of energy scale E.
    
    Parameters:
    - E: Energy scale (eV)
    - a: Offset parameter
    - b: Scaling factor for logarithmic dependence

    Returns:
    - λ_c (convergence rate) as a function of E
    """
    return a + b * np.log(1 + E)

# --- Hypothetical Model for Transaction Density ρ_T as a Function of Energy Scale ---
def rho_T_model(E, c, d):
    """
    Model for transaction density ρ_T as a function of energy scale E.

    Parameters:
    - E: Energy scale (eV)
    - c: Scaling coefficient
    - d: Power-law exponent

    Returns:
    - ρ_T (transaction density) as a function of E
    """
    return c * E**d

# --- Generate Simulated Data for λ_c and ρ_T ---
# Theoretical values based on assumed logarithmic and power-law relationships.
lambda_c_data = -0.1 + 0.01 * np.log(1 + energy_scale)  # Simulated λ_c values
rho_T_data = 0.8 * energy_scale**0.5  # Simulated ρ_T values (square-root dependence)

# --- Perform Curve Fitting for Both Models ---
popt_lambda, _ = curve_fit(lambda_c_model, energy_scale, lambda_c_data)  # Fit λ_c model
popt_rho, _ = curve_fit(rho_T_model, energy_scale, rho_T_data)  # Fit ρ_T model

# --- Plot the Fitting Results ---
plt.figure(figsize=(10, 5))

# --- Plot for λ_c vs. Energy Scale ---
plt.subplot(1, 2, 1)
plt.plot(energy_scale, lambda_c_data, 'r.', label="Simulated Data")  # Scatter plot of simulated data
plt.plot(energy_scale, lambda_c_model(energy_scale, *popt_lambda), 'g-', 
         label=f"Fitted Model: λ_c(E) = {popt_lambda[0]:.4f} + {popt_lambda[1]:.4f} log(1 + E)")
plt.xscale('log')  # Logarithmic x-axis
plt.xlabel("Energy Scale (eV)")
plt.ylabel("λ_c (Convergence Rate)")
plt.title("λ_c vs. Energy Scale")
plt.legend()

# --- Plot for ρ_T vs. Energy Scale ---
plt.subplot(1, 2, 2)
plt.plot(energy_scale, rho_T_data, 'b.', label="Simulated Data")  # Scatter plot of simulated data
plt.plot(energy_scale, rho_T_model(energy_scale, *popt_rho), 'g-', 
         label=f"Fitted Model: ρ_T(E) = {popt_rho[0]:.4f} * E^{popt_rho[1]:.4f}")
plt.xscale('log')  # Logarithmic x-axis
plt.yscale('log')  # Logarithmic y-axis
plt.xlabel("Energy Scale (eV)")
plt.ylabel("ρ_T (Transaction Density)")
plt.title("ρ_T vs. Energy Scale")
plt.legend()

plt.tight_layout()
plt.show()

# --- Output the Optimized Parameters ---
print("\nOptimized Parameters for λ_c Model:")
print(f"Offset (a) = {popt_lambda[0]:.4f}")
print(f"Scaling Factor (b) = {popt_lambda[1]:.4f}")

print("\nOptimized Parameters for ρ_T Model:")
print(f"Scaling Coefficient (c) = {popt_rho[0]:.4f}")
print(f"Power-Law Exponent (d) = {popt_rho[1]:.4f}")

# Return optimized fitting parameters
popt_lambda, popt_rho
