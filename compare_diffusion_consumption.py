import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    D,
    Vmax,
    Km,
    rho_cell,
    C0,
    C_boundary,
    Lx,
    Ly,
    dx,
    dy,
    dt,
    T
)

from model import (
    diffusion,
    diffusion_consumption_model
)


# ==================================
# 1. Create the 2D grid
# ==================================

Nx = int(Lx / dx) + 1
Ny = int(Ly / dy) + 1

centre_x = Nx // 2
centre_y = Ny // 2

radius = 15

print("Grid size:", Nx, "x", Ny)


# ==================================
# 2. Function to create
#    the same initial state
# ==================================

def create_initial_state():

    C = np.full((Nx, Ny), C0)

    # Lower-oxygen circular wound region
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


# ==================================
# 3. Create two identical models
# ==================================

C_diffusion = create_initial_state()

C_consumption = create_initial_state()


# ==================================
# 4. Storage for centre oxygen
# ==================================

time_values = []

centre_diffusion = []

centre_consumption = []


# ==================================
# 5. Run both simulations
# ==================================

steps = int(T / dt)

print("Simulation steps:", steps)


for step in range(steps):

    # ------------------------------
    # A. Diffusion only
    # ------------------------------

    dCdt_diffusion = diffusion(
        C_diffusion,
        D,
        dx,
        dy
    )

    C_diffusion = (
        C_diffusion
        + dt * dCdt_diffusion
    )


    # ------------------------------
    # B. Diffusion + consumption
    # ------------------------------

    dCdt_consumption = (
        diffusion_consumption_model(
            C_consumption,
            D,
            Vmax,
            Km,
            rho_cell,
            dx,
            dy
        )
    )

    C_consumption = (
        C_consumption
        + dt * dCdt_consumption
    )


    # ------------------------------
    # Reapply boundary conditions
    # ------------------------------

    C_diffusion[0, :] = C_boundary
    C_diffusion[-1, :] = C_boundary
    C_diffusion[:, 0] = C_boundary
    C_diffusion[:, -1] = C_boundary

    C_consumption[0, :] = C_boundary
    C_consumption[-1, :] = C_boundary
    C_consumption[:, 0] = C_boundary
    C_consumption[:, -1] = C_boundary


    # Prevent negative oxygen values
    C_diffusion = np.maximum(
        C_diffusion,
        0
    )

    C_consumption = np.maximum(
        C_consumption,
        0
    )


    # ------------------------------
    # Save centre oxygen over time
    # ------------------------------

    if step % 100 == 0:

        current_time = step * dt

        time_values.append(
            current_time
        )

        centre_diffusion.append(
            C_diffusion[
                centre_x,
                centre_y
            ]
        )

        centre_consumption.append(
            C_consumption[
                centre_x,
                centre_y
            ]
        )


# ==================================
# 6. Print final results
# ==================================

print("Simulation complete")

print(
    "Final centre oxygen - diffusion only:",
    C_diffusion[
        centre_x,
        centre_y
    ]
)

print(
    "Final centre oxygen - diffusion + consumption:",
    C_consumption[
        centre_x,
        centre_y
    ]
)


# ==================================
# 7. Plot comparison over time
# ==================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    time_values,
    centre_diffusion,
    label="Diffusion only"
)

plt.plot(
    time_values,
    centre_consumption,
    label="Diffusion + consumption"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Centre oxygen concentration (mol/m³)"
)

plt.title(
    "Effect of Cellular Oxygen Consumption"
)

plt.legend()

plt.grid()

plt.show()