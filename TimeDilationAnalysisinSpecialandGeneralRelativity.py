#　TimeDilationAnalysisinSpecialandGeneralRelativity


# --- Import Required Libraries ---
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Define Time Dilation Functions ---

def time_dilation_sr(T, v, c=3e8):
    """
    Computes time dilation due to velocity (Special Relativity).

    Parameters:
    - T: Proper time (reference time in stationary frame)
    - v: Velocity of moving observer (m/s)
    - c: Speed of light (default: 3e8 m/s)

    Returns:
    - Dilated time in the moving frame
    """
    gamma = 1 / np.sqrt(1 - (v / c)**2)  # Lorentz factor
    return T * gamma  # Time dilation correction

def time_dilation_gr(T, r, Rs):
    """
    Computes gravitational time dilation (General Relativity).

    Parameters:
    - T: Proper time (reference time in a weak gravitational field)
    - r: Radial distance from the black hole center (multiples of Schwarzschild radius)
    - Rs: Schwarzschild radius

    Returns:
    - Dilated time in the strong gravitational field
    """
    return T / np.sqrt(1 - Rs / r)  # General relativistic correction

# --- Define Key Parameters ---

T_133Cs = 1 / (9.192631770 * 10**9)  # Cs-133 atomic clock cycle (seconds)
v_values = np.linspace(0, 0.9 * 3e8, 100)  # Velocity range (0 to 0.9c)
r_values = np.linspace(2, 10, 100)  # Schwarzschild radius multiples (r/Rs)
Rs = 1  # Schwarzschild radius (relative units)

# --- Compute Time Dilation Values ---
T_sr_values = [time_dilation_sr(T_133Cs, v) for v in v_values]  # Special Relativity
T_gr_values = [time_dilation_gr(T_133Cs, r, Rs) for r in r_values]  # General Relativity

# --- Plot: Time Dilation in Special Relativity ---
plt.figure(figsize=(8, 5))
plt.plot(v_values / 3e8, T_sr_values, label="Time Dilation (Special Relativity)", color='red')
plt.xlabel("Velocity (v / c)")
plt.ylabel("Dilated Time (T')")
plt.title("Time Dilation in Special Relativity (Cs-133 Standard)")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot: Time Dilation in General Relativity ---
plt.figure(figsize=(8, 5))
plt.plot(r_values, T_gr_values, label="Time Dilation (General Relativity)", color='blue')
plt.xlabel("Schwarzschild Radius Multiples (r / Rs)")
plt.ylabel("Dilated Time (T')")
plt.title("Time Dilation in General Relativity (Cs-133 Standard)")
plt.legend()
plt.grid(True)
plt.show()

# --- Convert Computed Data to DataFrames ---
df_time_dilation_sr = pd.DataFrame({
    "Velocity (v / c)": v_values / 3e8,
    "Dilated Time (Special Relativity)": T_sr_values
})
df_time_dilation_gr = pd.DataFrame({
    "Radius (r / Rs)": r_values,
    "Dilated Time (General Relativity)": T_gr_values
})

# --- Display Computed Data ---
print("Special Relativity Time Dilation Data (First 10 Rows):")
print(df_time_dilation_sr.head(10))

print("\nGeneral Relativity Time Dilation Data (First 10 Rows):")
print(df_time_dilation_gr.head(10))
