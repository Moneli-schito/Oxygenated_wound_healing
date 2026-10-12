import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    D,
    C0,
    C_boundary,
    Lx,
    Ly,
    dx,
    dy,
    dt,
    T
)

from model import diffusion


# ==================================
# 1. Create the 2D grid
# ==================================

Nx = int(Lx / dx) + 1
Ny = int(Ly / dy) + 1

print("Grid size:", Nx, "x", Ny)


# ==================================
# 2. Initial oxygen concentration
# ==================================

C = np.full((Nx, Ny), C0)


# Create a lower-oxygen circular wound region
centre_x = Nx // 2
centre_y = Ny // 2

radius = 15

for i in range(Nx):
    for j in range(Ny):
        if (
            (i - centre_x) ** 2
            + (j - centre_y) ** 2
        ) < radius ** 2:
            C[i, j] = 0.05


# ==================================
# 3. Fixed boundary conditions
# ==================================

C[0, :] = C_boundary
C[-1, :] = C_boundary
C[:, 0] = C_boundary
C[:, -1] = C_boundary


# Save the initial oxygen distribution
C_initial = C.copy()


# ==================================
# 4. Run the simulation
# ==================================

steps = int(T / dt)

print("Simulation steps:", steps)


for step in range(steps):

    dCdt = diffusion(
        C,
        D,
        dx,
        dy
    )

    # Forward Euler update
    C = C + dt * dCdt

    # Reapply fixed boundary conditions
    C[0, :] = C_boundary
    C[-1, :] = C_boundary
    C[:, 0] = C_boundary
    C[:, -1] = C_boundary

    # Prevent very small negative values
    # caused by numerical error
    C = np.maximum(C, 0)


# ==================================
# 5. Print results
# ==================================

print("Simulation complete")

print(
    "Initial minimum oxygen:",
    C_initial.min()
)

print(
    "Final minimum oxygen:",
    C.min()
)

print(
    "Initial maximum oxygen:",
    C_initial.max()
)

print(
    "Final maximum oxygen:",
    C.max()
)


# ==================================
# 6. Plot initial vs final result
# ==================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


# Use the same colour scale
# for both plots
vmin = min(
    C_initial.min(),
    C.min()
)

vmax = max(
    C_initial.max(),
    C.max()
)


# ----------------------------------
# Initial oxygen distribution
# ----------------------------------

image1 = axes[0].imshow(
    C_initial,
    origin="lower",
    extent=[
        0,
        Lx * 1000,
        0,
        Ly * 1000
    ],
    vmin=vmin,
    vmax=vmax
)

axes[0].set_title(
    "Initial Oxygen Distribution"
)

axes[0].set_xlabel(
    "x position (mm)"
)

axes[0].set_ylabel(
    "y position (mm)"
)


# ----------------------------------
# Final oxygen distribution
# ----------------------------------

image2 = axes[1].imshow(
    C,
    origin="lower",
    extent=[
        0,
        Lx * 1000,
        0,
        Ly * 1000
    ],
    vmin=vmin,
    vmax=vmax
)

axes[1].set_title(
    f"After {T} s of Diffusion"
)

axes[1].set_xlabel(
    "x position (mm)"
)

axes[1].set_ylabel(
    "y position (mm)"
)


# ==================================
# 7. Shared colour bar
# ==================================

fig.colorbar(
    image2,
    ax=axes,
    label="Oxygen concentration (mol/m³)"
)


plt.suptitle(
    "Diffusion-Only Oxygen Simulation"
)

plt.show()