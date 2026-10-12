import numpy as np
import time

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
    T,
    beta_normal,
)

from model import full_oxygen_model


# ============================================
# GRID
# ============================================

nx = int(Lx / dx) + 1
ny = int(Ly / dy) + 1

centre_x = nx // 2
centre_y = ny // 2


# ============================================
# INITIAL CONDITION
# ============================================

def create_initial_state():

    C = np.full((nx, ny), C0)

    radius = 15

    y, x = np.ogrid[:nx, :ny]

    mask = (
        (x - centre_y) ** 2
        + (y - centre_x) ** 2
        <= radius ** 2
    )

    C[mask] = 0.05

    # Fixed boundaries
    C[0, :] = C_boundary
    C[-1, :] = C_boundary
    C[:, 0] = C_boundary
    C[:, -1] = C_boundary

    return C


# ============================================
# RUN MODEL WITH SELECTED dt
# ============================================

def run_model(dt_test):

    C = create_initial_state()

    n_steps = int(T / dt_test)

    hypoxia_exit_time = None

    start = time.perf_counter()

    for step in range(n_steps):

        dCdt = full_oxygen_model(
            C,
            D,
            Vmax,
            Km,
            rho_cell,
            beta_normal,
            Cb,
            dx,
            dy,
        )

        # Forward Euler
        C = C + dt_test * dCdt

        C = np.maximum(C, 0.0)

        # Fixed boundaries
        C[0, :] = C_boundary
        C[-1, :] = C_boundary
        C[:, 0] = C_boundary
        C[:, -1] = C_boundary

        # Exact threshold check at each time step
        if (
            hypoxia_exit_time is None
            and C[centre_x, centre_y] >= C_hyp
        ):
            hypoxia_exit_time = (
                (step + 1) * dt_test
            )

    end = time.perf_counter()

    final_centre = C[
        centre_x,
        centre_y
    ]

    runtime = end - start

    return (
        final_centre,
        hypoxia_exit_time,
        runtime,
        n_steps,
    )


# ============================================
# TEST TIME STEPS
# ============================================

dt_values = [
    0.01,
    0.05,
    0.10,
    0.20,
]


print("==============================================")
print("TIME-STEP CONVERGENCE / PERFORMANCE TEST")
print("==============================================")

print(
    "\nGrid:",
    nx,
    "x",
    ny
)

print(
    "Simulation duration:",
    T,
    "seconds"
)


results = []


for dt_test in dt_values:

    print(
        "\nRunning dt =",
        dt_test,
        "s..."
    )

    final_centre, exit_time, runtime, steps = (
        run_model(dt_test)
    )

    results.append(
        (
            dt_test,
            final_centre,
            exit_time,
            runtime,
            steps,
        )
    )


# ============================================
# REFERENCE RESULT
# ============================================

reference_oxygen = results[0][1]


# ============================================
# PRINT RESULTS
# ============================================

print("\n")
print(
    "=============================================================="
)

print(
    "dt       Steps      Final O2       Exit time     Runtime"
)

print(
    "=============================================================="
)


for (
    dt_test,
    final_centre,
    exit_time,
    runtime,
    steps,
) in results:

    print(
        f"{dt_test:<8}"
        f"{steps:<11}"
        f"{final_centre:<15.10f}"
        f"{exit_time:<14}"
        f"{runtime:.3f} s"
    )


# ============================================
# ERROR RELATIVE TO dt = 0.01
# ============================================

print("\n")
print(
    "=============================================="
)

print(
    "ERROR RELATIVE TO dt = 0.01 s"
)

print(
    "=============================================="
)


for (
    dt_test,
    final_centre,
    exit_time,
    runtime,
    steps,
) in results:

    absolute_error = abs(
        final_centre - reference_oxygen
    )

    percent_error = (
        absolute_error
        / reference_oxygen
        * 100
    )

    print(
        f"dt = {dt_test:<5} "
        f"Final O2 error = "
        f"{percent_error:.6f}%"
    )


# ============================================
# ESTIMATED SOBOL TIME
# ============================================

print("\n")
print(
    "=============================================="
)

print(
    "ESTIMATED TIME FOR 448 SOBOL RUNS"
)

print(
    "=============================================="
)


for (
    dt_test,
    final_centre,
    exit_time,
    runtime,
    steps,
) in results:

    estimated_minutes = (
        runtime * 448 / 60
    )

    print(
        f"dt = {dt_test:<5} "
        f"Estimated Sobol time = "
        f"{estimated_minutes:.2f} minutes"
    )