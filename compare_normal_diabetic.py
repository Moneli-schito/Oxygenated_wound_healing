
import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    D,
    Vmax,
    Km,
    rho_cell,
    beta_normal,
    beta_diabetic,
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
# NORMAL VS DIABETIC WOUND OXYGENATION
# Conservative Zero-Flux Boundary
# ============================================


# --------------------------------------------
# 1. Computational grid and wound geometry
# --------------------------------------------

Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1

centre_x = Nx // 2
centre_y = Ny // 2

steps = int(round(T / dt))
times = np.arange(steps + 1) * dt

wound_mask = create_wound_mask(
    Nx,
    Ny,
    radius=15
)

wound_perfusion_fraction = 0.5

print("Grid:", Nx, "x", Ny)
print("Wound-bed cells:", np.sum(wound_mask))
print("Simulation duration:", T, "seconds")
print("Time steps:", steps)
print("Boundary condition: Conservative zero-flux")


# --------------------------------------------
# 2. Numerical stability check
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
# 3. Simulation function
# --------------------------------------------

def simulate_condition(name, beta_surrounding):

    beta_field = create_perfusion_field(
        wound_mask,
        beta_surrounding=beta_surrounding,
        wound_perfusion_fraction=wound_perfusion_fraction
    )

    C = np.full(
        (Nx, Ny),
        C0,
        dtype=float
    )

    C[wound_mask] = 0.05

    centre_history = np.empty(steps + 1)
    wound_mean_history = np.empty(steps + 1)
    hypoxia_history = np.empty(steps + 1)

    centre_history[0] = C[centre_x, centre_y]
    wound_mean_history[0] = np.mean(C[wound_mask])

    hypoxia_history[0] = (
        np.mean(C[wound_mask] < C_hyp) * 100
    )

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

        C = C + dt * dCdt

        if np.any(C < 0):
            raise ValueError(
                name + ": Negative oxygen concentration."
            )

        centre_history[step + 1] = (
            C[centre_x, centre_y]
        )

        wound_mean_history[step + 1] = (
            np.mean(C[wound_mask])
        )

        hypoxia_history[step + 1] = (
            np.mean(C[wound_mask] < C_hyp) * 100
        )

    crossing_indices = np.flatnonzero(
        centre_history >= C_hyp
    )

    if len(crossing_indices) > 0:
        centre_exit_time = times[crossing_indices[0]]
    else:
        centre_exit_time = None

    return {
        "name": name,
        "beta_surrounding": beta_surrounding,
        "beta_wound": beta_field[centre_x, centre_y],
        "final_field": C,
        "centre_history": centre_history,
        "wound_mean_history": wound_mean_history,
        "hypoxia_history": hypoxia_history,
        "centre_exit_time": centre_exit_time,
        "final_centre": C[centre_x, centre_y],
        "mean_wound": np.mean(C[wound_mask]),
        "mean_surrounding": np.mean(C[~wound_mask]),
        "final_hypoxia": hypoxia_history[-1]
    }


# --------------------------------------------
# 4. Run both conditions
# --------------------------------------------

normal = simulate_condition(
    "Normal",
    beta_normal
)

diabetic = simulate_condition(
    "Diabetic",
    beta_diabetic
)


# --------------------------------------------
# 5. Print comparative results
# --------------------------------------------

print("\n" + "=" * 60)
print("NORMAL VS DIABETIC RESULTS")
print("=" * 60)

for result in [normal, diabetic]:

    print("\nCondition:", result["name"])

    print(
        "Surrounding perfusion:",
        result["beta_surrounding"],
        "1/s"
    )

    print(
        "Wound-bed perfusion:",
        result["beta_wound"],
        "1/s"
    )

    print(
        "Final wound-centre pO2:",
        concentration_to_pO2(result["final_centre"]),
        "mmHg"
    )

    print(
        "Mean wound-bed pO2:",
        concentration_to_pO2(result["mean_wound"]),
        "mmHg"
    )

    print(
        "Mean surrounding pO2:",
        concentration_to_pO2(result["mean_surrounding"]),
        "mmHg"
    )

    print(
        "Final hypoxic wound area:",
        result["final_hypoxia"],
        "%"
    )

    print(
        "Centre threshold crossing:",
        result["centre_exit_time"],
        "seconds"
    )


difference = concentration_to_pO2(
    normal["mean_wound"] - diabetic["mean_wound"]
)

print("\nMean wound-bed pO2 difference:", difference, "mmHg")
print("Provisional hypoxia threshold:", pO2_hyp, "mmHg")


# --------------------------------------------
# 6. Comparative oxygen heatmaps
# --------------------------------------------

normal_pO2 = concentration_to_pO2(
    normal["final_field"]
)

diabetic_pO2 = concentration_to_pO2(
    diabetic["final_field"]
)

vmin = min(
    np.min(normal_pO2),
    np.min(diabetic_pO2)
)

vmax = max(
    np.max(normal_pO2),
    np.max(diabetic_pO2)
)

extent = [0, Lx * 1000, 0, Ly * 1000]

x_mm = np.linspace(0, Lx * 1000, Nx)
y_mm = np.linspace(0, Ly * 1000, Ny)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
    constrained_layout=True
)

for ax, field, title in zip(
    axes,
    [normal_pO2, diabetic_pO2],
    [
        "Normal Wound - Final pO2",
        "Diabetic Wound - Final pO2"
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
    "Normal vs Diabetic Wound Oxygenation"
)

plt.show()


# --------------------------------------------
# 7. Compare wound-centre oxygen recovery
# --------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    times,
    concentration_to_pO2(normal["centre_history"]),
    label="Normal wound centre"
)

plt.plot(
    times,
    concentration_to_pO2(diabetic["centre_history"]),
    label="Diabetic wound centre",
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
plt.title("Normal vs Diabetic Wound-Centre Oxygen Recovery")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# --------------------------------------------
# 8. Compare hypoxic wound area
# --------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    times,
    normal["hypoxia_history"],
    label="Normal"
)

plt.plot(
    times,
    diabetic["hypoxia_history"],
    label="Diabetic",
    linestyle="--"
)

plt.xlabel("Time (s)")
plt.ylabel("Hypoxic wound-bed area (%)")
plt.title("Normal vs Diabetic Hypoxic Wound Area")
plt.ylim(-5, 105)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
