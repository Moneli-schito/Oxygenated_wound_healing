
import numpy as np
import time

from parameters import (
    D,
    Vmax,
    Km,
    rho_cell,
    Cb,
    C0,
    C_hyp,
    Lx,
    Ly,
    dx,
    dy,
    beta_normal,
    beta_diabetic,
    alpha_O2,
)

from model import (
    full_oxygen_model,
    create_wound_mask,
    create_perfusion_field,
)


# ============================================
# SOBOL TIMESTEP ACCURACY TEST
# ============================================

# Same physical simulation duration for every test
T = 600.0

# Candidate timesteps in seconds
TIMESTEPS = [0.1, 0.01, 0.005]

# Exploratory wound perfusion reduction
WOUND_PERFUSION_FRACTION = 0.5

# Numerical comparison tolerances
CENTRE_TOLERANCE_PERCENT = 0.1
MEAN_TOLERANCE_PERCENT = 0.1
EXIT_TIME_TOLERANCE_SECONDS = 0.2


# ============================================
# GRID AND INITIAL CONDITION
# ============================================

Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1

centre_x = Nx // 2
centre_y = Ny // 2

wound_mask = create_wound_mask(
    Nx,
    Ny,
    radius=15
)


def create_initial_state():
    C = np.full(
        (Nx, Ny),
        C0,
        dtype=float
    )

    C[wound_mask] = 0.05

    return C


# ============================================
# RUN SIMULATION
# ============================================

def run_simulation(beta_surrounding, dt):
    C = create_initial_state()

    beta_field = create_perfusion_field(
        wound_mask,
        beta_surrounding,
        wound_perfusion_fraction=WOUND_PERFUSION_FRACTION
    )

    n_steps = int(round(T / dt))

    exit_time = None

    start = time.perf_counter()

    for step in range(n_steps):
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
            raise RuntimeError(
                f"Non-finite concentration at "
                f"step {step + 1}, dt={dt}"
            )

        if np.min(C) < -1e-10:
            raise RuntimeError(
                f"Negative concentration at "
                f"step {step + 1}, dt={dt}"
            )

        C = np.maximum(C, 0.0)

        if (
            exit_time is None
            and C[centre_x, centre_y] >= C_hyp
        ):
            exit_time = (step + 1) * dt

    runtime = time.perf_counter() - start

    centre_pO2 = (
        C[centre_x, centre_y] / alpha_O2
    )

    mean_wound_pO2 = (
        np.mean(C[wound_mask]) / alpha_O2
    )

    final_hypoxic_fraction = (
        np.mean(C[wound_mask] < C_hyp) * 100
    )

    return {
        "centre_pO2": centre_pO2,
        "mean_wound_pO2": mean_wound_pO2,
        "exit_time": exit_time,
        "hypoxic_fraction": final_hypoxic_fraction,
        "runtime": runtime,
    }


# ============================================
# COMPARE RESULTS
# ============================================

def compare_condition(condition_name, beta_value):
    print("\n" + "=" * 70)
    print(f"{condition_name.upper()} CONDITION")
    print("=" * 70)

    results = {}

    for dt in TIMESTEPS:
        print(f"\nRunning dt = {dt} s ...")

        results[dt] = run_simulation(
            beta_value,
            dt
        )

        r = results[dt]

        print(
            f"Centre pO2: {r['centre_pO2']:.6f} mmHg"
        )
        print(
            f"Mean wound pO2: "
            f"{r['mean_wound_pO2']:.6f} mmHg"
        )
        print(
            f"Hypoxia exit time: "
            f"{r['exit_time']} s"
        )
        print(
            f"Final hypoxic area: "
            f"{r['hypoxic_fraction']:.4f}%"
        )
        print(
            f"Runtime: {r['runtime']:.2f} s"
        )

    reference = results[0.005]
    candidate = results[0.1]

    centre_error = (
        abs(
            candidate["centre_pO2"]
            - reference["centre_pO2"]
        )
        / max(abs(reference["centre_pO2"]), 1e-12)
        * 100
    )

    mean_error = (
        abs(
            candidate["mean_wound_pO2"]
            - reference["mean_wound_pO2"]
        )
        / max(abs(reference["mean_wound_pO2"]), 1e-12)
        * 100
    )

    if (
        candidate["exit_time"] is not None
        and reference["exit_time"] is not None
    ):
        exit_error = abs(
            candidate["exit_time"]
            - reference["exit_time"]
        )
    elif (
        candidate["exit_time"] is None
        and reference["exit_time"] is None
    ):
        exit_error = 0.0
    else:
        exit_error = float("inf")

    hypoxic_error = abs(
        candidate["hypoxic_fraction"]
        - reference["hypoxic_fraction"]
    )

    print("\n" + "-" * 70)
    print("COMPARISON: dt = 0.1 s vs dt = 0.005 s")
    print("-" * 70)

    print(
        f"Centre pO2 relative difference: "
        f"{centre_error:.6f}%"
    )
    print(
        f"Mean wound pO2 relative difference: "
        f"{mean_error:.6f}%"
    )
    print(
        f"Hypoxia exit time difference: "
        f"{exit_error:.6f} s"
    )
    print(
        f"Final hypoxic area difference: "
        f"{hypoxic_error:.6f} percentage points"
    )

    passed = (
        centre_error <= CENTRE_TOLERANCE_PERCENT
        and mean_error <= MEAN_TOLERANCE_PERCENT
        and exit_error <= EXIT_TIME_TOLERANCE_SECONDS
        and hypoxic_error <= 0.1
    )

    print(
        "\nRESULT:",
        "PASS" if passed else "REVIEW REQUIRED"
    )

    return passed


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("REVISED WOUND MODEL - SOBOL TIMESTEP TEST")
    print("=" * 70)

    print(f"Grid: {Nx} x {Ny}")
    print(f"Simulation duration: {T} s")
    print("Boundary condition: zero flux")
    print(
        f"Wound perfusion fraction: "
        f"{WOUND_PERFUSION_FRACTION}"
    )
    print(
        "Hypoxia threshold: provisional, "
        "not clinically validated"
    )

    normal_pass = compare_condition(
        "Normal",
        beta_normal
    )

    diabetic_pass = compare_condition(
        "Diabetic",
        beta_diabetic
    )

    print("\n" + "=" * 70)
    print("OVERALL TIMESTEP TEST")
    print("=" * 70)

    if normal_pass and diabetic_pass:
        print(
            "PASS: dt = 0.1 s meets the selected "
            "comparison tolerances for both "
            "baseline conditions."
        )
    else:
        print(
            "REVIEW REQUIRED: dt = 0.1 s does not "
            "meet all selected comparison tolerances."
        )

    print(
        "\nNote: Passing this test does not prove "
        "accuracy across the entire Sobol "
        "parameter space."
    )
