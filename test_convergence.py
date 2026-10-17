
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

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
    concentration_to_pO2
)

from model import (
    full_oxygen_model,
    create_perfusion_field
)


# ============================================
# NUMERICAL CONVERGENCE TEST
# Normal vs Diabetic Wound Oxygenation
# ============================================

# Use a shorter duration initially to make
# the convergence test computationally practical.
#
# This is a 100-second convergence test.
# It does not replace the 600-second simulation.

T_test = 100.0

wound_radius_m = 0.0015
wound_perfusion_fraction = 0.5
initial_wound_oxygen = 0.05


# --------------------------------------------
# 1. Define numerical configurations
# --------------------------------------------

configurations = [
    {
        "name": "Baseline",
        "dx": 1e-4,
        "dy": 1e-4,
        "dt": 0.01
    },
    {
        "name": "Finer time",
        "dx": 1e-4,
        "dy": 1e-4,
        "dt": 0.005
    },
    {
        "name": "Finer grid",
        "dx": 5e-5,
        "dy": 5e-5,
        "dt": 0.01
    },
    {
        "name": "Finer grid + time",
        "dx": 5e-5,
        "dy": 5e-5,
        "dt": 0.005
    }
]


# --------------------------------------------
# 2. Simulation function
# --------------------------------------------

def run_simulation(beta_surrounding, dx, dy, dt):

    Nx = int(round(Lx / dx)) + 1
    Ny = int(round(Ly / dy)) + 1

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    X, Y = np.meshgrid(x, y, indexing="ij")

    centre_x_m = Lx / 2
    centre_y_m = Ly / 2

    wound_mask = (
        (X - centre_x_m)**2
        + (Y - centre_y_m)**2
    ) < wound_radius_m**2

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

    C[wound_mask] = initial_wound_oxygen

    diffusion_number = (
        D * dt * (1 / dx**2 + 1 / dy**2)
    )

    if diffusion_number > 0.5:
        raise ValueError(
            "Explicit diffusion stability condition failed."
        )

    steps = int(round(T_test / dt))

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

        if not np.all(np.isfinite(C)):
            raise ValueError(
                "Non-finite oxygen concentration detected."
            )

        if np.any(C < 0):
            raise ValueError(
                "Negative oxygen concentration detected."
            )

    centre_x = Nx // 2
    centre_y = Ny // 2

    return {
        "centre_pO2": concentration_to_pO2(
            C[centre_x, centre_y]
        ),
        "mean_wound_pO2": concentration_to_pO2(
            np.mean(C[wound_mask])
        ),
        "hypoxic_area": (
            np.mean(C[wound_mask] < C_hyp) * 100
        ),
        "grid": f"{Nx} x {Ny}",
        "diffusion_number": diffusion_number
    }


# --------------------------------------------
# 3. Run convergence experiments
# --------------------------------------------

results = []

print("\nNUMERICAL CONVERGENCE TEST")
print("Duration:", T_test, "seconds")
print("Wound radius:", wound_radius_m * 1000, "mm")
print("=" * 70)

for config in configurations:

    start = perf_counter()

    normal = run_simulation(
        beta_normal,
        config["dx"],
        config["dy"],
        config["dt"]
    )

    diabetic = run_simulation(
        beta_diabetic,
        config["dx"],
        config["dy"],
        config["dt"]
    )

    elapsed = perf_counter() - start

    difference = (
        normal["mean_wound_pO2"]
        - diabetic["mean_wound_pO2"]
    )

    result = {
        "name": config["name"],
        "dx_mm": config["dx"] * 1000,
        "dt": config["dt"],
        "grid": normal["grid"],
        "normal_centre": normal["centre_pO2"],
        "diabetic_centre": diabetic["centre_pO2"],
        "normal_mean": normal["mean_wound_pO2"],
        "diabetic_mean": diabetic["mean_wound_pO2"],
        "difference": difference,
        "elapsed": elapsed
    }

    results.append(result)

    print("\nConfiguration:", result["name"])
    print("Grid:", result["grid"])
    print("dx:", result["dx_mm"], "mm")
    print("dt:", result["dt"], "seconds")

    print(
        "Normal centre pO2:",
        round(result["normal_centre"], 4),
        "mmHg"
    )

    print(
        "Diabetic centre pO2:",
        round(result["diabetic_centre"], 4),
        "mmHg"
    )

    print(
        "Normal mean wound pO2:",
        round(result["normal_mean"], 4),
        "mmHg"
    )

    print(
        "Diabetic mean wound pO2:",
        round(result["diabetic_mean"], 4),
        "mmHg"
    )

    print(
        "Normal-diabetic difference:",
        round(result["difference"], 4),
        "mmHg"
    )

    print(
        "Execution time:",
        round(result["elapsed"], 2),
        "seconds"
    )


# --------------------------------------------
# 4. Compare against finest configuration
# --------------------------------------------

reference = results[-1]

print("\n" + "=" * 70)
print("CONVERGENCE COMPARISON")
print("=" * 70)

for result in results:

    centre_error_normal = abs(
        result["normal_centre"]
        - reference["normal_centre"]
    )

    centre_error_diabetic = abs(
        result["diabetic_centre"]
        - reference["diabetic_centre"]
    )

    difference_error = abs(
        result["difference"]
        - reference["difference"]
    )

    print("\n", result["name"])

    print(
        "Normal centre deviation:",
        round(centre_error_normal, 6),
        "mmHg"
    )

    print(
        "Diabetic centre deviation:",
        round(centre_error_diabetic, 6),
        "mmHg"
    )

    print(
        "Difference deviation:",
        round(difference_error, 6),
        "mmHg"
    )


# --------------------------------------------
# 5. Plot convergence results
# --------------------------------------------

labels = [r["name"] for r in results]

normal_values = [
    r["normal_centre"] for r in results
]

diabetic_values = [
    r["diabetic_centre"] for r in results
]

x = np.arange(len(labels))

plt.figure(figsize=(10, 5))

plt.plot(
    x,
    normal_values,
    marker="o",
    label="Normal wound centre"
)

plt.plot(
    x,
    diabetic_values,
    marker="s",
    label="Diabetic wound centre"
)

plt.xticks(x, labels, rotation=15)

plt.ylabel("Wound-centre pO2 (mmHg)")
plt.title(
    "Numerical Convergence at 100 Seconds"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# --------------------------------------------
# 6. Plot normal-diabetic difference
# --------------------------------------------

differences = [
    r["difference"] for r in results
]

plt.figure(figsize=(10, 5))

plt.plot(
    x,
    differences,
    marker="o"
)

plt.xticks(x, labels, rotation=15)

plt.ylabel(
    "Normal - diabetic mean wound pO2 (mmHg)"
)

plt.title(
    "Convergence of Perfusion-Related Difference"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
