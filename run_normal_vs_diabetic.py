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
    T,
    beta_normal,
    beta_diabetic,
)

from model import full_oxygen_model


# ============================================
# GRID SETUP
# ============================================

nx = int(Lx / dx) + 1
ny = int(Ly / dy) + 1

n_steps = int(T / dt)

print("Grid size:", nx, "x", ny)
print("Simulation steps:", n_steps)
print("Normal beta:", beta_normal, "1/s")
print("Diabetic beta:", beta_diabetic, "1/s")


# ============================================
# INITIAL CONDITION
# ============================================

def create_initial_state():

    C = np.full((nx, ny), C0)

    # Centre of the wound domain
    centre_x = nx // 2
    centre_y = ny // 2

    # Radius = 15 grid cells = 1.5 mm
    radius = 15

    for i in range(nx):
        for j in range(ny):

            distance = np.sqrt(
                (i - centre_x) ** 2
                + (j - centre_y) ** 2
            )

            if distance <= radius:
                C[i, j] = 0.05

    # Fixed boundary conditions
    C[0, :] = C_boundary
    C[-1, :] = C_boundary
    C[:, 0] = C_boundary
    C[:, -1] = C_boundary

    return C


# ============================================
# SIMULATION FUNCTION
# ============================================

def run_simulation(beta_value):

    C = create_initial_state()

    centre_x = nx // 2
    centre_y = ny // 2

    centre_history = []
    time_history = []

    for step in range(n_steps):

        dCdt = full_oxygen_model(
            C,
            D,
            Vmax,
            Km,
            rho_cell,
            beta_value,
            Cb,
            dx,
            dy,
        )

        # Forward Euler update
        C = C + dt * dCdt

        # Oxygen concentration cannot be negative
        C = np.maximum(C, 0.0)

        # Reapply fixed boundary conditions
        C[0, :] = C_boundary
        C[-1, :] = C_boundary
        C[:, 0] = C_boundary
        C[:, -1] = C_boundary

        # Save centre oxygen every 100 steps
        # 100 steps x 0.01 s = 1 second
        if step % 100 == 0:

            centre_history.append(
                C[centre_x, centre_y]
            )

            time_history.append(
                (step + 1) * dt
            )

    return (
        C,
        np.array(time_history),
        np.array(centre_history),
    )


# ============================================
# RUN NORMAL MODEL
# ============================================

print("\nRunning NORMAL simulation...")

C_normal, time_normal, centre_normal = run_simulation(
    beta_normal
)

print("Normal simulation complete.")


# ============================================
# RUN DIABETIC MODEL
# ============================================

print("\nRunning DIABETIC simulation...")

C_diabetic, time_diabetic, centre_diabetic = run_simulation(
    beta_diabetic
)

print("Diabetic simulation complete.")


# ============================================
# CALCULATE FINAL RESULTS
# ============================================

centre_x = nx // 2
centre_y = ny // 2

normal_centre = C_normal[
    centre_x,
    centre_y
]

diabetic_centre = C_diabetic[
    centre_x,
    centre_y
]

normal_min = np.min(
    C_normal
)

diabetic_min = np.min(
    C_diabetic
)

normal_hypoxic_fraction = (
    np.mean(C_normal < C_hyp) * 100
)

diabetic_hypoxic_fraction = (
    np.mean(C_diabetic < C_hyp) * 100
)


# ============================================
# TIME TO EXIT HYPOXIA
# ============================================

def time_to_exit_hypoxia(
    time_history,
    oxygen_history,
    threshold
):

    # Find all recorded times where centre oxygen
    # is equal to or greater than the threshold
    above_threshold = np.where(
        oxygen_history >= threshold
    )[0]

    # If oxygen never reaches the threshold
    if len(above_threshold) == 0:
        return None

    # First point where oxygen reaches/exceeds
    # the hypoxia threshold
    first_index = above_threshold[0]

    return time_history[first_index]


normal_exit_time = time_to_exit_hypoxia(
    time_normal,
    centre_normal,
    C_hyp
)

diabetic_exit_time = time_to_exit_hypoxia(
    time_diabetic,
    centre_diabetic,
    C_hyp
)


# ============================================
# CALCULATE REOXYGENATION DELAY
# ============================================

if (
    normal_exit_time is not None
    and diabetic_exit_time is not None
):

    reoxygenation_delay = (
        diabetic_exit_time
        - normal_exit_time
    )

else:

    reoxygenation_delay = None


# ============================================
# PRINT RESULTS
# ============================================

print("\n======================================")
print("NORMAL VS DIABETIC RESULTS")
print("======================================")


print("\nNORMAL MODEL")

print(
    "Beta:",
    beta_normal,
    "1/s"
)

print(
    "Final centre oxygen:",
    normal_centre
)

print(
    "Final minimum oxygen:",
    normal_min
)

print(
    "Hypoxic tissue fraction:",
    normal_hypoxic_fraction,
    "%"
)


print("\nDIABETIC MODEL")

print(
    "Beta:",
    beta_diabetic,
    "1/s"
)

print(
    "Final centre oxygen:",
    diabetic_centre
)

print(
    "Final minimum oxygen:",
    diabetic_min
)

print(
    "Hypoxic tissue fraction:",
    diabetic_hypoxic_fraction,
    "%"
)


# ============================================
# PRINT HYPOXIA RECOVERY RESULTS
# ============================================

print("\n======================================")
print("TIME TO EXIT HYPOXIA")
print("======================================")

if normal_exit_time is not None:

    print(
        "Normal:",
        normal_exit_time,
        "seconds"
    )

else:

    print(
        "Normal: Did not exit hypoxia "
        "during simulation"
    )


if diabetic_exit_time is not None:

    print(
        "Diabetic:",
        diabetic_exit_time,
        "seconds"
    )

else:

    print(
        "Diabetic: Did not exit hypoxia "
        "during simulation"
    )


if reoxygenation_delay is not None:

    print(
        "Diabetic reoxygenation delay:",
        reoxygenation_delay,
        "seconds"
    )


# ============================================
# FIGURE 1
# CENTRE OXYGEN VS TIME
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    time_normal,
    centre_normal,
    label="Normal"
)

plt.plot(
    time_diabetic,
    centre_diabetic,
    label="Diabetic / impaired perfusion"
)

plt.axhline(
    C_hyp,
    linestyle="--",
    label="Hypoxia threshold"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Centre oxygen concentration (mol/m³)"
)

plt.title(
    "Normal vs Diabetic Wound Oxygenation"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================
# SHARED COLOUR SCALE FOR HEATMAPS
# ============================================

# Using the same colour scale makes the normal
# and diabetic heatmaps directly comparable.

heatmap_min = min(
    np.min(C_normal),
    np.min(C_diabetic)
)

heatmap_max = max(
    np.max(C_normal),
    np.max(C_diabetic)
)


# ============================================
# FIGURE 2
# NORMAL OXYGEN HEATMAP
# ============================================

plt.figure(figsize=(6, 5))

plt.imshow(
    C_normal.T,
    origin="lower",
    extent=[
        0,
        Lx * 1000,
        0,
        Ly * 1000
    ],
    aspect="equal",
    vmin=heatmap_min,
    vmax=heatmap_max
)

plt.colorbar(
    label="Oxygen concentration (mol/m³)"
)

plt.xlabel(
    "x (mm)"
)

plt.ylabel(
    "y (mm)"
)

plt.title(
    "Normal Tissue - Oxygen Distribution at 600 s"
)

plt.tight_layout()

plt.show()


# ============================================
# FIGURE 3
# DIABETIC OXYGEN HEATMAP
# ============================================

plt.figure(figsize=(6, 5))

plt.imshow(
    C_diabetic.T,
    origin="lower",
    extent=[
        0,
        Lx * 1000,
        0,
        Ly * 1000
    ],
    aspect="equal",
    vmin=heatmap_min,
    vmax=heatmap_max
)

plt.colorbar(
    label="Oxygen concentration (mol/m³)"
)

plt.xlabel(
    "x (mm)"
)

plt.ylabel(
    "y (mm)"
)

plt.title(
    "Diabetic Tissue - Oxygen Distribution at 600 s"
)

plt.tight_layout()

plt.show()


# ============================================
# FIGURE 4
# SPATIAL PROFILE THROUGH WOUND CENTRE
# ============================================

x_mm = np.linspace(
    0,
    Lx * 1000,
    nx
)

normal_profile = C_normal[
    :,
    centre_y
]

diabetic_profile = C_diabetic[
    :,
    centre_y
]


plt.figure(figsize=(8, 5))

plt.plot(
    x_mm,
    normal_profile,
    label="Normal"
)

plt.plot(
    x_mm,
    diabetic_profile,
    label="Diabetic / impaired perfusion"
)

plt.axhline(
    C_hyp,
    linestyle="--",
    label="Hypoxia threshold"
)

plt.xlabel(
    "Distance across wound (mm)"
)

plt.ylabel(
    "Oxygen concentration (mol/m³)"
)

plt.title(
    "Spatial Oxygen Profile Through Wound Centre"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()