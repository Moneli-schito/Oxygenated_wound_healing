
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
    C_boundary,
    C_hyp,
    Lx,
    Ly,
    dx,
    dy,
    dt,
    T
)

from model import (
    full_oxygen_model,
    create_wound_mask,
    create_perfusion_field
)


# ============================================
# EXPERIMENT: THREE BOUNDARY CONDITIONS
# ============================================

Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1

centre = (Nx // 2, Ny // 2)
steps = int(round(T / dt))

wound_mask = create_wound_mask(
    Nx,
    Ny,
    radius=15
)

# Exploratory wound-bed perfusion reduction
wound_fraction = 0.5

beta_field = create_perfusion_field(
    wound_mask,
    beta_surrounding=beta_normal,
    wound_perfusion_fraction=wound_fraction
)

cases = {
    "A: Fixed 0.091": {
        "type": "dirichlet",
        "value": C_boundary
    },
    "B: Fixed 0.135": {
        "type": "dirichlet",
        "value": 0.135
    },
    "C: Zero-flux": {
        "type": "neumann",
        "value": None
    }
}


# ============================================
# BOUNDARY CONDITION FUNCTIONS
# ============================================

def apply_boundary(C, boundary_type, value=None):
    """
    Apply either:
    - Fixed-concentration (Dirichlet)
    - Zero-gradient (Neumann) boundary
    """

    if boundary_type == "dirichlet":

        C[0, :] = value
        C[-1, :] = value
        C[:, 0] = value
        C[:, -1] = value

    elif boundary_type == "neumann":

        # Copy adjacent interior concentrations
        # to enforce zero normal gradient.

        C[0, :] = C[1, :]
        C[-1, :] = C[-2, :]
        C[:, 0] = C[:, 1]
        C[:, -1] = C[:, -2]

    else:
        raise ValueError(
            "Unknown boundary condition type."
        )

    return C


# ============================================
# RUN SIMULATION
# ============================================

def run_simulation(boundary_type, boundary_value):

    C = np.full(
        (Nx, Ny),
        C0,
        dtype=float
    )

    C[wound_mask] = 0.05

    apply_boundary(
        C,
        boundary_type,
        boundary_value
    )

    initial = C.copy()

    centre_history = np.empty(steps + 1)
    centre_history[0] = C[centre]

    for step in range(steps):

        # The existing Laplacian updates interior
        # grid cells only. Boundary values are
        # imposed after each time step.

        dCdt = full_oxygen_model(
            C,
            D,
            Vmax,
            Km,
            rho_cell,
            beta_field,
            Cb,
            dx,
            dy
        )

        C = C + dt * dCdt

        C = np.maximum(C, 0.0)

        apply_boundary(
            C,
            boundary_type,
            boundary_value
        )

        centre_history[step + 1] = C[centre]

    return initial, C, centre_history


# ============================================
# RUN ALL THREE CASES
# ============================================

results = {}

for name, settings in cases.items():

    print("\nRunning:", name)

    initial, final, history = run_simulation(
        settings["type"],
        settings["value"]
    )

    wound_mean = np.mean(
        final[wound_mask]
    )

    surrounding_mean = np.mean(
        final[~wound_mask]
    )

    wound_hypoxia = (
        np.mean(
            final[wound_mask] < C_hyp
        ) * 100
    )

    results[name] = {
        "initial": initial,
        "final": final,
        "history": history,
        "wound_mean": wound_mean,
        "surrounding_mean": surrounding_mean,
        "wound_hypoxia": wound_hypoxia
    }

    print(
        "Final centre oxygen:",
        final[centre]
    )

    print(
        "Mean wound oxygen:",
        wound_mean
    )

    print(
        "Mean surrounding oxygen:",
        surrounding_mean
    )

    print(
        "Wound hypoxic fraction:",
        wound_hypoxia,
        "%"
    )


# ============================================
# FIGURE 1: FINAL OXYGEN HEATMAPS
# ============================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    constrained_layout=True
)

final_fields = [
    result["final"]
    for result in results.values()
]

vmin = min(
    np.min(field)
    for field in final_fields
)

vmax = max(
    np.max(field)
    for field in final_fields
)

extent = [
    0,
    Lx * 1000,
    0,
    Ly * 1000
]

x_mm = np.linspace(
    0,
    Lx * 1000,
    Nx
)

y_mm = np.linspace(
    0,
    Ly * 1000,
    Ny
)

for ax, (name, result) in zip(
    axes,
    results.items()
):

    image = ax.imshow(
        result["final"].T,
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

    ax.set_title(name)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

fig.colorbar(
    image,
    ax=axes,
    label="Oxygen concentration (mol/m³)",
    shrink=0.8
)

fig.suptitle(
    "Comparison of Three Boundary Conditions"
)

plt.show()


# ============================================
# FIGURE 2: WOUND-CENTRE OXYGEN OVER TIME
# ============================================

times = np.arange(steps + 1) * dt

plt.figure(figsize=(10, 5))

for name, result in results.items():

    plt.plot(
        times,
        result["history"],
        label=name
    )

plt.axhline(
    C_hyp,
    linestyle="--",
    color="red",
    label="Working hypoxia threshold"
)

plt.xlabel("Time (s)")
plt.ylabel("Oxygen concentration (mol/m³)")
plt.title("Wound-Centre Oxygen Recovery")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# ============================================
# FIGURE 3: WOUND VS SURROUNDING TISSUE
# ============================================

names = list(results.keys())

wound_means = [
    results[name]["wound_mean"]
    for name in names
]

surrounding_means = [
    results[name]["surrounding_mean"]
    for name in names
]

x = np.arange(len(names))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(
    x - width / 2,
    wound_means,
    width,
    label="Wound bed"
)

plt.bar(
    x + width / 2,
    surrounding_means,
    width,
    label="Surrounding tissue"
)

plt.xticks(x, names)
plt.ylabel("Mean oxygen concentration (mol/m³)")
plt.title("Wound vs Surrounding Tissue Oxygen")
plt.legend()
plt.tight_layout()

plt.show()
