
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

# Retained for compatibility with earlier
# fixed-boundary simulations.
# Not used in the revised zero-flux model.

C_boundary = 0.091      # mol/m^3

# Provisional working hypoxia threshold.
# Not yet validated as a clinical wound threshold.

C_hyp = 0.061           # mol/m^3


# --------------------------------------------
# 5. OXYGEN PARTIAL PRESSURE CONVERSION
# --------------------------------------------

# Henry's law:
#
#     C = alpha * pO2
#
# Therefore:
#
#     pO2 = C / alpha
#
# C: oxygen concentration (mol/m^3)
# pO2: oxygen partial pressure (mmHg)
# alpha: oxygen solubility (mol/(m^3 * mmHg))
#
# This provisional coefficient approximates
# oxygen solubility in plasma at 37 degrees C.
#
# It is NOT a validated skin wound-tissue
# solubility coefficient.
#
# Tissue-specific literature validation is
# required before physiological interpretation.

alpha_O2 = 1.26e-3      # mol/(m^3 * mmHg)


def concentration_to_pO2(C):
    """
    Convert dissolved oxygen concentration
    (mol/m^3) to partial pressure (mmHg).

    Works with scalars and NumPy arrays.
    """

    return C / alpha_O2


def pO2_to_concentration(pO2):
    """
    Convert oxygen partial pressure (mmHg)
    to dissolved concentration (mol/m^3).

    Works with scalars and NumPy arrays.
    """

    return alpha_O2 * pO2


# Derived reference values.
# These are not independent clinical thresholds.

pO2_hyp = concentration_to_pO2(C_hyp)
pO2_blood = concentration_to_pO2(Cb)
pO2_initial = concentration_to_pO2(C0)


# --------------------------------------------
# 6. 2D WOUND DOMAIN
# --------------------------------------------

Lx = 0.01               # Domain width = 10 mm

Ly = 0.01               # Domain height = 10 mm


# --------------------------------------------
# 7. SPATIAL RESOLUTION
# --------------------------------------------

dx = 1e-4               # Grid spacing in x = 0.1 mm

dy = 1e-4               # Grid spacing in y = 0.1 mm


# --------------------------------------------
# 8. TIME SETTINGS
# --------------------------------------------

dt = 0.01               # Time step (seconds)

T = 600.0               # Duration = 10 minutes


# --------------------------------------------
# 9. PERFUSION PARAMETERS
# --------------------------------------------

# Perfusion equation:
#
#     P(C) = beta * (Cb - C)
#
# beta has units of s^-1.
#
# These effective perfusion coefficients are
# provisional and require further validation.

beta_normal = 6.9e-3       # s^-1

beta_diabetic = 4.77e-3    # s^-1


# --------------------------------------------
# 10. DEFAULT PERFUSION VALUE
# --------------------------------------------

# Preserve compatibility with previous scripts.

beta = beta_normal
