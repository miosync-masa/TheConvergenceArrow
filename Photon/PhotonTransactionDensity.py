import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
c = 3.0e8  # Speed of light (m/s)
alpha = 1/137  # Fine-structure constant (dimensionless)
m_e = 9.10938356e-31  # Electron mass (kg)

# --- Define Energy Scale ---
E = np.logspace(-2, 2, 100)  # Energy range from 0.01 eV to 100 eV

# --- 1. Photon Transaction Density (ρ_T) ---
def transaction_density(E, hbar):
    """
    Computes the transaction density for photons based on the energy scale.

    Parameters:
    - E: Energy scale [eV]
    - hbar: Reduced Planck's constant [J·s]

    Returns:
    - ρ_T: Photon transaction density
    """
    return E / hbar

rho_T = transaction_density(E, hbar)

# --- 2. Entanglement Entropy Decay (S_ent) ---
def entanglement_entropy(E, lambda_c, N_T, time, hbar):
    """
    Models entanglement entropy decay as a function of energy.

    Parameters:
    - E: Energy scale [eV]
    - lambda_c: Interaction coefficient
    - N_T: Number of transactions
    - time: Time scale in seconds
    - hbar: Reduced Planck's constant [J·s]

    Returns:
    - S_ent: Entanglement entropy
    """
    return np.exp(-lambda_c * N_T / (E / hbar) * time)

lambda_c = 0.1  # Interaction coefficient
N_T = 1e5  # Number of transactions
time = 1e-9  # Time scale in seconds
S_ent = entanglement_entropy(E, lambda_c, N_T, time, hbar)

# --- 3. QED Self-Interaction via Transaction Density ---
def qed_transaction_density(alpha, m_e, c, E):
    """
    Computes the QED self-interaction transaction density.

    Parameters:
    - alpha: Fine-structure constant
    - m_e: Electron mass [kg]
    - c: Speed of light [m/s]
    - E: Energy scale [eV]

    Returns:
    - ρ_T_QED: QED transaction density
    """
    lambda_C = hbar / (m_e * c)  # Compton wavelength
    rho_T_QED = (alpha**2) / lambda_C**3  # Base QED transaction density
    rho_T_QED_corrected = rho_T_QED * (1 + 0.1 * np.sin(2 * np.pi * E / 10))  # Oscillatory correction
    return rho_T_QED_corrected

rho_T_QED_corrected = qed_transaction_density(alpha, m_e, c, E)

# --- Plot Results ---
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# --- Plot: Transaction Density vs Energy Scale ---
axs[0].plot(E, rho_T, label=r"$\rho_T = \frac{E}{\hbar}$", color='b')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel("Energy Scale (E) [eV]")
axs[0].set_ylabel(r"Transaction Density $\rho_T$")
axs[0].set_title("Photon Transaction Density vs. Energy Scale")
axs[0].legend()
axs[0].grid()

# --- Plot: Entanglement Entropy Decay ---
axs[1].plot(E, S_ent, label=r"$S_{ent} = e^{-\lambda_c N_T / (E / \hbar) t}$", color='r')
axs[1].set_xscale('log')
axs[1].set_xlabel("Energy Scale (E) [eV]")
axs[1].set_ylabel("Entanglement Entropy $S_{ent}$")
axs[1].set_title("Entanglement Decay over Energy Scale")
axs[1].legend()
axs[1].grid()

# --- Plot: QED Self-Interaction Transaction Density ---
axs[2].plot(E, rho_T_QED_corrected, label=r"$\rho_T' = \rho_T (1 + 0.1 \sin(2\pi E / 10))$", color='g')
axs[2].set_xscale('log')
axs[2].set_xlabel("Energy Scale (E) [eV]")
axs[2].set_ylabel("Transaction Density (QED)")
axs[2].set_title("QED Self-Interaction via Transaction Density")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()
