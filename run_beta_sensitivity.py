import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    D,
    Vmax,
    Km,
    rho_cell,
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
# EXPERIMENT 4
# Perfusion Sensitivity Analysis
# ============================================

Nx = int(Lx / dx) + 1
Ny = int(Ly / dy) + 1

centre_x = Nx // 2
centre_y = Ny // 2

radius = 15

# Test different effective perfusion coefficients
beta_values = [
    0.001725,
    0.00345,
    0.0069,
    0.01035
]


def create_initial_state():
    """
    Create the same initial oxygen distribution
    for every beta simulation.
    """

    C = np.full((Nx, Ny), C0)

    for i in range(Nx):
        for j in range(Ny):

            if (
                (i - centre_x) ** 2
                + (j - centre_y) ** 2
            ) < radius ** 2:

                C[i, j] = 0.05

    # Fixed boundary conditions
    C[0, :] = C_boundary
    C[-1, :] = C_boundary
    C[:, 0] = C_boundary
    C[:, -1] = C_boundary

    return C


# Store results
final_centre_values = []
hypoxic_fractions = []


print("Grid size:", Nx, "x", Ny)
print("Simulation time:", T, "seconds")
print()


# ============================================
# Run one simulation for each beta
# ============================================

for beta_test in beta_values:

    print(
        "Running beta =",
        beta_test,
        "1/s"
    )

    C = create_initial_state()

    steps = int(T / dt)

    for step in range(steps):

        dCdt = full_oxygen_model(
            C,
            D,
            Vmax,
            Km,
            rho_cell,
            beta_test,
            Cb,
            dx,
            dy
        )

        # Forward Euler update
        C = C + dt * dCdt

        # Fixed boundary conditions
        C[0, :] = C_boundary
        C[-1, :] = C_boundary
        C[:, 0] = C_boundary
        C[:, -1] = C_boundary

        # Prevent negative concentrations
        C = np.maximum(C, 0)

    # Final centre oxygen
    centre_value = C[
        centre_x,
        centre_y
    ]

    # Percentage of domain below hypoxia threshold
    hypoxic_fraction = (
        np.sum(C < C_hyp)
        / C.size
        * 100
    )

    final_centre_values.append(
        centre_value
    )

    hypoxic_fractions.append(
        hypoxic_fraction
    )

    print(
        "Final centre oxygen:",
        centre_value
    )

    print(
        "Hypoxic fraction:",
        hypoxic_fraction,
        "%"
    )

    print()


# ============================================
# Print summary
# ============================================

print("----- SUMMARY -----")

for i in range(len(beta_values)):

    print(
        "beta =",
        beta_values[i],
        "| centre O2 =",
        final_centre_values[i],
        "| hypoxic fraction =",
        hypoxic_fractions[i],
        "%"
    )


# ============================================
# Plot beta vs centre oxygen
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    beta_values,
    final_centre_values,
    marker="o"
)

# Show hypoxia threshold
plt.axhline(
    y=C_hyp,
    linestyle="--",
    label="Hypoxia threshold"
)

plt.xlabel(
    "Perfusion coefficient beta (1/s)"
)

plt.ylabel(
    "Final centre oxygen concentration (mol/m³)"
)

plt.title(
    "Sensitivity of Wound Oxygen to Perfusion"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()