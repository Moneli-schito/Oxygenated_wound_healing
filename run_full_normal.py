import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    D,
    Vmax,
    Km,
    rho_cell,
    beta,
    Cb,
    C0,
    C_boundary,
    C_hyp,
    Lx,
    Ly,
    dx,
    dy,
    dt,
    T
)

from model import full_oxygen_model


# ============================================
# EXPERIMENT 3
# Full Normal Wound Oxygen Model
#
# Diffusion - Consumption + Perfusion
# ============================================


# --------------------------------------------
# 1. Create the 2D grid
# --------------------------------------------

Nx = int(Lx / dx) + 1
Ny = int(Ly / dy) + 1

print("Grid size:", Nx, "x", Ny)


# --------------------------------------------
# 2. Initial oxygen concentration
# --------------------------------------------

C = np.full((Nx, Ny), C0)

centre_x = Nx // 2
centre_y = Ny // 2

# Circular low-oxygen wound region
radius = 15

for i in range(Nx):
    for j in range(Ny):

        if (
            (i - centre_x) ** 2
            + (j - centre_y) ** 2
        ) < radius ** 2:

            C[i, j] = 0.05


# --------------------------------------------
# 3. Apply fixed boundary conditions
# --------------------------------------------

C[0, :] = C_boundary
C[-1, :] = C_boundary
C[:, 0] = C_boundary
C[:, -1] = C_boundary


# Save initial state for plotting
C_initial = C.copy()


# --------------------------------------------
# 4. Time simulation
# --------------------------------------------

steps = int(T / dt)

print("Simulation steps:", steps)
print("Perfusion coefficient beta:", beta, "1/s")

for step in range(steps):

    # Calculate rate of oxygen change
    dCdt = full_oxygen_model(
        C,
        D,
        Vmax,
        Km,
        rho_cell,
        beta,
        Cb,
        dx,
        dy
    )

    # Forward Euler time update
    C = C + dt * dCdt

    # Reapply fixed boundary conditions
    C[0, :] = C_boundary
    C[-1, :] = C_boundary
    C[:, 0] = C_boundary
    C[:, -1] = C_boundary

    # Prevent negative oxygen concentrations
    C = np.maximum(C, 0)


# --------------------------------------------
# 5. Simulation results
# --------------------------------------------

print("Simulation complete")

print(
    "Initial centre oxygen:",
    C_initial[centre_x, centre_y]
)

print(
    "Final centre oxygen:",
    C[centre_x, centre_y]
)

print(
    "Final minimum oxygen:",
    np.min(C)
)

print(
    "Final maximum oxygen:",
    np.max(C)
)

print(
    "Hypoxia threshold:",
    C_hyp
)


# Percentage of grid below hypoxia threshold
hypoxic_fraction = (
    np.sum(C < C_hyp)
    / C.size
    * 100
)

print(
    "Hypoxic tissue fraction:",
    hypoxic_fraction,
    "%"
)


# --------------------------------------------
# 6. Plot initial and final oxygen maps
# --------------------------------------------

vmin = min(
    np.min(C_initial),
    np.min(C)
)

vmax = max(
    np.max(C_initial),
    np.max(C)
)


plt.figure(figsize=(12, 5))


# Initial oxygen map
plt.subplot(1, 2, 1)

plt.imshow(
    C_initial.T,
    origin="lower",
    extent=[0, Lx * 1000, 0, Ly * 1000],
    vmin=vmin,
    vmax=vmax,
    aspect="equal"
)

plt.colorbar(
    label="Oxygen concentration (mol/m³)"
)

plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("Initial Oxygen Distribution")


# Final oxygen map
plt.subplot(1, 2, 2)

plt.imshow(
    C.T,
    origin="lower",
    extent=[0, Lx * 1000, 0, Ly * 1000],
    vmin=vmin,
    vmax=vmax,
    aspect="equal"
)

plt.colorbar(
    label="Oxygen concentration (mol/m³)"
)

plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title(
    "Full Normal Model\n"
    "Diffusion + Consumption + Perfusion"
)


plt.tight_layout()
plt.show()