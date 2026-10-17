
import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    D,
    Vmax,
    Km,
    rho_cell,
    beta_normal,
    Cb,
    C0,
    C_hyp,
    Lx,
    Ly,
    dx,
    dy,
    dt,
    T,
    concentration_to_pO2,
    pO2_hyp
)

from model import (
    full_oxygen_model,
    create_wound_mask,
    create_perfusion_field
)


# ============================================
# NORMAL WOUND OXYGEN MODEL
# Conservative Zero-Flux Boundary
# Oxygen Partial Pressure Visualisation
# ============================================


# --------------------------------------------
# 1. Create computational grid
# --------------------------------------------

Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1

centre_x = Nx // 2
centre_y = Ny // 2

steps = int(round(T / dt))

print("Grid size:", Nx, "x", Ny)
print("Simulation steps:", steps)


# --------------------------------------------
# 2. Define wound-bed geometry
# --------------------------------------------

wound_radius = 15

wound_mask = create_wound_mask(
    Nx,
    Ny,
    radius=wound_radius
)

print("Wound-bed cells:", np.sum(wound_mask))


# --------------------------------------------
# 3. Create spatial perfusion field
# --------------------------------------------

# Exploratory assumption:
# Wound perfusion is 50% of surrounding tissue.

wound_perfusion_fraction = 0.5

beta_field = create_perfusion_field(
    wound_mask,
    beta_surrounding=beta_normal,
    wound_perfusion_fraction=wound_perfusion_fraction
)

print("Surrounding perfusion:", beta_normal, "1/s")
print(
    "Wound-bed perfusion:",
    beta_field[centre_x, centre_y],
    "1/s"
)


# --------------------------------------------
# 4. Set initial oxygen concentration
# --------------------------------------------

C = np.full(
    (Nx, Ny),
    C0,
    dtype=float
)

C[wound_mask] = 0.05

C_initial = C.copy()


# --------------------------------------------
# 5. Numerical stability check
# --------------------------------------------

diffusion_number = (
    D * dt * (1 / dx**2 + 1 / dy**2)
)

if diffusion_number > 0.5:
    raise ValueError(
        "Time step is too large for explicit diffusion."
    )

print("Diffusion stability number:", diffusion_number)


# --------------------------------------------
# 6. Run full oxygen simulation
# --------------------------------------------

centre_history = np.empty(steps + 1)
centre_history[0] = C[centre_x, centre_y]

wound_mean_history = np.empty(steps + 1)
wound_mean_history[0] = np.mean(C[wound_mask])

hypoxia_history = np.empty(steps + 1)
hypoxia_history[0] = (
    np.mean(C[wound_mask] < C_hyp) * 100
)

times = np.arange(steps + 1) * dt

for step in range(steps):

    dCdt = full_oxygen_model(
        C,
        D,
        Vmax,
        Km,
        rho_cell,
        beta_field,
        Cb,
        dx,
        dy,
        boundary_type="zero_flux"
    )

    # Forward Euler update
    C = C + dt * dCdt

    # Do not overwrite the boundary values.
    # Zero-flux is handled by the Laplacian.

    centre_history[step + 1] = C[centre_x, centre_y]
    wound_mean_history[step + 1] = np.mean(C[wound_mask])

    hypoxia_history[step + 1] = (
        np.mean(C[wound_mask] < C_hyp) * 100
    )


# --------------------------------------------
# 7. Convert concentration to pO2
# --------------------------------------------

pO2_initial = concentration_to_pO2(C_initial)
pO2_final = concentration_to_pO2(C)

centre_pO2_history = concentration_to_pO2(
    centre_history
)

wound_mean_pO2_history = concentration_to_pO2(
    wound_mean_history
)


# --------------------------------------------
# 8. Calculate final results
# --------------------------------------------

wound_values = C[wound_mask]
surrounding_values = C[~wound_mask]

final_centre = C[centre_x, centre_y]

mean_wound = np.mean(wound_values)
mean_surrounding = np.mean(surrounding_values)

final_wound_hypoxia = (
    np.mean(wound_values < C_hyp) * 100
)

above_threshold = np.flatnonzero(
    centre_history >= C_hyp
)

if len(above_threshold) > 0:
    centre_exit_time = times[above_threshold[0]]
else:
    centre_exit_time = None


print("\nSIMULATION COMPLETE")

print("Boundary condition: Conservative zero-flux")

print("\nCONCENTRATION RESULTS")
print("Initial centre oxygen:", C_initial[centre_x, centre_y])
print("Final centre oxygen:", final_centre)
print("Mean wound-bed oxygen:", mean_wound)
print("Mean surrounding oxygen:", mean_surrounding)

print("\nPARTIAL PRESSURE RESULTS")
print(
    "Initial centre pO2:",
    concentration_to_pO2(C_initial[centre_x, centre_y]),
    "mmHg"
)
print(
    "Final centre pO2:",
    concentration_to_pO2(final_centre),
    "mmHg"
)
print(
    "Mean wound-bed pO2:",
    concentration_to_pO2(mean_wound),
    "mmHg"
)
print(
    "Mean surrounding pO2:",
    concentration_to_pO2(mean_surrounding),
    "mmHg"
)

print("\nHYPOXIA RESULTS")
print("Working hypoxia threshold:", pO2_hyp, "mmHg")
print("Initial hypoxic wound area:", hypoxia_history[0], "%")
print("Final hypoxic wound area:", final_wound_hypoxia, "%")

if centre_exit_time is not None:
    print(
        "Wound-centre threshold crossing time:",
        centre_exit_time,
        "seconds"
    )
else:
    print("Wound centre remained below threshold.")


# --------------------------------------------
# 9. Plot oxygen partial pressure heatmaps
# --------------------------------------------

extent = [
    0,
    Lx * 1000,
    0,
    Ly * 1000
]

x_mm = np.linspace(0, Lx * 1000, Nx)
y_mm = np.linspace(0, Ly * 1000, Ny)

vmin = min(
    np.min(pO2_initial),
    np.min(pO2_final)
)

vmax = max(
    np.max(pO2_initial),
    np.max(pO2_final)
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
    constrained_layout=True
)

for ax, field, title in zip(
    axes,
    [pO2_initial, pO2_final],
    [
        "Initial Oxygen Partial Pressure",
        "Final Oxygen Partial Pressure"
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

    ax.contour(
        x_mm,
        y_mm,
        wound_mask.T.astype(float),
        levels=[0.5],
        colors="white",
        linewidths=1.5
    )

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

fig.colorbar(
    image,
    ax=axes,
    label="Oxygen partial pressure (mmHg)",
    shrink=0.8
)

fig.suptitle(
    "Normal Wound Oxygenation - Zero-Flux Model"
)

plt.show()


# --------------------------------------------
# 10. Plot oxygen recovery in mmHg
# --------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    times,
    centre_pO2_history,
    label="Wound-centre pO2"
)

plt.plot(
    times,
    wound_mean_pO2_history,
    label="Mean wound-bed pO2",
    linestyle="--"
)

plt.axhline(
    pO2_hyp,
    color="red",
    linestyle=":",
    label="Provisional hypoxia threshold"
)

plt.xlabel("Time (s)")
plt.ylabel("Oxygen partial pressure (mmHg)")
plt.title("Wound Oxygen Partial Pressure Over Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# --------------------------------------------
# 11. Plot wound hypoxic area over time
# --------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    times,
    hypoxia_history
)

plt.xlabel("Time (s)")
plt.ylabel("Hypoxic wound-bed area (%)")
plt.title("Wound-Bed Hypoxic Area Over Time")
plt.ylim(-5, 105)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
