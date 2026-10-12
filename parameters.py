# ============================================
# OXYGENATED WOUND HEALING MODEL
# Model Parameters
# ============================================


# --------------------------------------------
# 1. OXYGEN DIFFUSION
# --------------------------------------------

D = 1.7e-9              # Oxygen diffusion coefficient (m^2/s)


# --------------------------------------------
# 2. OXYGEN CONSUMPTION
# --------------------------------------------

Vmax = 1.3e-17          # Maximum O2 consumption rate (mol/cell/s)

Km = 8.0e-3             # Michaelis-Menten constant (mol/m^3)


# --------------------------------------------
# 3. CELL DENSITY
# --------------------------------------------

# Literature value = 2.85e6 cells/mL
# Converted to SI units:
rho_cell = 2.85e12      # cells/m^3


# --------------------------------------------
# 4. OXYGEN CONCENTRATIONS
# --------------------------------------------

Cb = 0.144              # Blood/source oxygen concentration (mol/m^3)

C0 = 0.091              # Initial tissue oxygen concentration (mol/m^3)

C_boundary = 0.091      # Fixed boundary oxygen concentration (mol/m^3)

C_hyp = 0.061           # Working hypoxia threshold (mol/m^3)


# --------------------------------------------
# 5. 2D WOUND DOMAIN
# --------------------------------------------

Lx = 0.01               # Domain width = 10 mm = 0.01 m

Ly = 0.01               # Domain height = 10 mm = 0.01 m


# --------------------------------------------
# 6. SPATIAL RESOLUTION
# --------------------------------------------

dx = 1e-4               # Grid spacing in x = 0.1 mm

dy = 1e-4               # Grid spacing in y = 0.1 mm


# --------------------------------------------
# 7. TIME SETTINGS
# --------------------------------------------

dt = 0.01               # Time step (seconds)

T = 600.0               # Total simulation duration
                        # 600 seconds = 10 minutes


# --------------------------------------------
# 8. PERFUSION PARAMETERS
# --------------------------------------------

# Perfusion is represented in the model as:
#
#     P(C) = beta * (Cb - C)
#
# beta therefore has units of s^-1.
#
# beta_normal is the effective baseline perfusion
# coefficient used for the normal tissue model.
#
# beta_diabetic represents reduced microvascular
# perfusion in the initial diabetic-state model.
#
# The diabetic value is currently treated as a
# provisional modelling value and should be refined
# as additional literature/supervisor feedback is
# incorporated.

beta_normal = 6.9e-3       # Normal perfusion coefficient (s^-1)

beta_diabetic = 4.77e-3    # Diabetic/impaired perfusion coefficient (s^-1)


# --------------------------------------------
# 9. DEFAULT PERFUSION VALUE
# --------------------------------------------

# Keep beta as the normal value so that previous
# simulation scripts that import "beta" continue
# to work without modification.

beta = beta_normal