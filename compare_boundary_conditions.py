
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
# BOUNDARY CONDITION COMPARISON
# ============================================

Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1

wound_mask = create_wound_mask(
    Nx, Ny, radius=15
)

# Hypothetical wound perfusion reduction
wound_fraction = 0.5

beta_field = create_perfusion_field(
    wound_mask,
    beta_surrounding=beta_normal,
    wound_perfusion_fraction=wound_fraction
)

# Existing and exploratory boundary values
boundary_cases = {
    "A: Existing boundary": C_boundary,
    "B: Higher boundary (test)": 0.135
}

steps = int(round(T / dt))
centre = (Nx // 2, Ny // 2)

results = {}


def run_simulation(boundary_value):
    """
    Simulate oxygen transport using a fixed
    Dirichlet boundary concentration.
    """

    C = np.full(
        (Nx, Ny), C0, dtype=float
    )

    C[wound_mask] = 0.05

    # Apply boundary conditions
    C[0, :] = boundary_value
    C[-1, :] = boundary_value
    C[:, 0] = boundary_value
    C[:, -1] = boundary_value

    initial = C.copy()

    centre_history = np.empty(steps + 1)
    centre_history[0] = C[centre]

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
            dy
        )

        C = C + dt * dCdt

        C = np.maximum(C, 0.0)

        # Reapply fixed boundaries
        C[0, :] = boundary_value
        C[-1, :] = boundary_value
        C[:, 0] = boundary_value
        C[:, -1] = boundary_value

        centre_history[step + 1] = C[centre]

    return initial, C, centre_history


# ============================================
# RUN BOTH SCENARIOS
# ============================================

for name, boundary_value in boundary_cases.items():

    print("\nRunning:", name)

    initial, final, history = run_simulation(
        boundary_value
    )

    wound_mean = np.mean(final[wound_mask])
    surrounding_mean = np.mean(final[~wound_mask])

    wound_hypoxia = (
        np.mean(final[wound_mask] < C_hyp) * 100
    )

    results[name] = {
        "initial": initial,
        "final": final,
        "history": history,
        "wound_mean": wound_mean,
        "surrounding_mean": surrounding_mean,
        "wound_hypoxia": wound_hypoxia
    }

    print("Boundary concentration:", boundary_value)
    print("Final centre oxygen:", final[centre])
    print("Mean wound oxygen:", wound_mean)
    print("Mean surrounding oxygen:", surrounding_mean)
    print("Wound hypoxic fraction:", wound_hypoxia, "%")


# ============================================
# FIGURE 1: FINAL OXYGEN HEATMAPS
# ============================================

fig, axes = plt.subplots(
    1, 2, figsize=(12, 5),
    constrained_layout=True
)

all_final = [
    result["final"]
    for result in results.values()
]

vmin = min(np.min(a) for a in all_final)
vmax = max(np.max(a) for a in all_final)

extent = [0, Lx * 1000, 0, Ly * 1000]

x_mm = np.linspace(0, Lx * 1000, Nx)
y_mm = np.linspace(0, Ly * 1000, Ny)

for ax, (name, result) in zip(
    axes, results.items()
):

    im = ax.imshow(
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
    im,
    ax=axes,
    label="Oxygen concentration (mol/m³)",
    shrink=0.8
)

fig.suptitle(
    "Boundary Condition Sensitivity: Final Oxygen Distribution"
)

plt.show()


# ============================================
# FIGURE 2: WOUND-CENTRE RECOVERY
# ============================================

times = np.arange(steps + 1) * dt

plt.figure(figsize=(9, 5))

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
plt.ylabel("Wound-centre oxygen (mol/m³)")
plt.title("Boundary Condition Sensitivity: Wound Recovery")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# ============================================
# FIGURE 3: WOUND VS SURROUNDING MEAN
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

plt.figure(figsize=(9, 5))

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
