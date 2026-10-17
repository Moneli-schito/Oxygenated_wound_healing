
import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    D,
    Lx,
    Ly,
    dx,
    dy
)


# ============================================
# ZERO-FLUX BOUNDARY VERIFICATION
# ============================================

# This test uses diffusion only.
# No oxygen consumption or perfusion.


# --------------------------------------------
# 1. Grid settings
# --------------------------------------------

Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1

test_dt = 0.1
test_T = 600.0

steps = int(round(test_T / test_dt))

# Explicit 2D diffusion stability check
stability_number = (
    D * test_dt
    * (1 / dx**2 + 1 / dy**2)
)

if stability_number > 0.5:
    raise ValueError(
        "Time step violates the explicit "
        "diffusion stability condition."
    )

print("Grid:", Nx, "x", Ny)
print("Time step:", test_dt)
print("Simulation duration:", test_T)


# --------------------------------------------
# 2. Initial oxygen distribution
# --------------------------------------------

C = np.full(
    (Nx, Ny),
    0.091,
    dtype=float
)

# Central low-oxygen region
x, y = np.ogrid[:Nx, :Ny]

centre_x = Nx // 2
centre_y = Ny // 2

wound_mask = (
    (x - centre_x)**2
    + (y - centre_y)**2
) < 15**2

C[wound_mask] = 0.05

C_initial = C.copy()


# --------------------------------------------
# 3. Conservative zero-flux Laplacian
# --------------------------------------------

def zero_flux_laplacian(C, dx, dy):
    """
    Compute diffusion using zero-flux
    boundaries with a conservative
    finite-volume-style discretisation.

    No diffusive flux crosses the outer edges.
    """

    lap = np.zeros_like(C)

    # Flux contributions across internal
    # interfaces in the first direction.
    differences_x = (
        C[1:, :] - C[:-1, :]
    ) / dx**2

    lap[:-1, :] += differences_x
    lap[1:, :] -= differences_x

    # Flux contributions across internal
    # interfaces in the second direction.
    differences_y = (
        C[:, 1:] - C[:, :-1]
    ) / dy**2

    lap[:, :-1] += differences_y
    lap[:, 1:] -= differences_y

    return lap


# --------------------------------------------
# 4. Diffusion simulation
# --------------------------------------------

mass_initial = np.sum(C) * dx * dy

mass_history = np.empty(steps + 1)
mass_history[0] = mass_initial

centre_history = np.empty(steps + 1)
centre_history[0] = C[centre_x, centre_y]

for step in range(steps):

    lap = zero_flux_laplacian(
        C,
        dx,
        dy
    )

    C = C + test_dt * D * lap

    mass_history[step + 1] = (
        np.sum(C) * dx * dy
    )

    centre_history[step + 1] = (
        C[centre_x, centre_y]
    )


# --------------------------------------------
# 5. Verification results
# --------------------------------------------

mass_final = np.sum(C) * dx * dy

relative_mass_error = (
    abs(mass_final - mass_initial)
    / abs(mass_initial)
    * 100
)

print("\nZERO-FLUX VERIFICATION RESULTS")

print("Initial total oxygen:", mass_initial)
print("Final total oxygen:", mass_final)

print(
    "Relative mass conservation error:",
    relative_mass_error,
    "%"
)

print(
    "Initial centre oxygen:",
    C_initial[centre_x, centre_y]
)

print(
    "Final centre oxygen:",
    C[centre_x, centre_y]
)

print(
    "Minimum final concentration:",
    np.min(C)
)

print(
    "Maximum final concentration:",
    np.max(C)
)

if relative_mass_error < 1e-6:
    print("PASS: Total oxygen conserved.")
else:
    print("CHECK: Mass conservation error.")


# --------------------------------------------
# 6. Plot mass conservation
# --------------------------------------------

times = np.arange(steps + 1) * test_dt

plt.figure(figsize=(9, 5))

plt.plot(
    times,
    mass_history,
    label="Total oxygen"
)

plt.axhline(
    mass_initial,
    linestyle="--",
    color="red",
    label="Initial total oxygen"
)

plt.xlabel("Time (s)")
plt.ylabel("Integrated oxygen (mol/m)")
plt.title("Zero-Flux Diffusion: Mass Conservation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------
# 7. Plot oxygen redistribution
# --------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
    constrained_layout=True
)

vmin = min(
    np.min(C_initial),
    np.min(C)
)

vmax = max(
    np.max(C_initial),
    np.max(C)
)

extent = [
    0,
    Lx * 1000,
    0,
    Ly * 1000
]

for ax, field, title in zip(
    axes,
    [C_initial, C],
    [
        "Initial Oxygen Distribution",
        "Final Oxygen Distribution"
    ]
):

    image = ax.imshow(
        field.T,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        aspect="equal"
    )

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

fig.colorbar(
    image,
    ax=axes,
    label="Oxygen concentration (mol/m³)",
    shrink=0.8
)

plt.show()
