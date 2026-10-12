import numpy as np
import time
import matplotlib.pyplot as plt

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol

from parameters import (
    Cb,
    C0,
    C_boundary,
    C_hyp,
    Lx,
    Ly,
    dx,
    dy,
)

from model import full_oxygen_model


# ============================================
# SOBOL ANALYSIS SETTINGS
# ============================================

# Time step validated against dt = 0.01 s
# dt = 0.1 s produced only ~0.001% difference
# in final centre oxygen while greatly reducing runtime.
dt_sobol = 0.1

# Total simulation duration
T_sobol = 600.0

# Number of time steps
n_steps = int(T_sobol / dt_sobol)

# Sobol base sample size
# N = 256 gives:
# N * (D + 2)
# = 256 * (5 + 2)
# = 1792 model evaluations
N = 256


# ============================================
# GRID SETUP
# ============================================

nx = int(Lx / dx) + 1
ny = int(Ly / dy) + 1

centre_x = nx // 2
centre_y = ny // 2


# ============================================
# SOBOL PARAMETER SPACE
# ============================================

# Preliminary uncertainty ranges:
#
# D, Vmax, Km and rho_cell:
# +/- 20% around baseline values
#
# beta:
# 50% to 150% of normal baseline
#
# IMPORTANT:
# These are preliminary sensitivity-analysis
# ranges and should not be interpreted as
# definitive physiological uncertainty ranges.

problem = {

    "num_vars": 5,

    "names": [
        "D",
        "Vmax",
        "Km",
        "rho_cell",
        "beta",
    ],

    "bounds": [
        [1.36e-9, 2.04e-9],
        [1.04e-17, 1.56e-17],
        [6.40e-3, 9.60e-3],
        [2.28e12, 3.42e12],
        [3.45e-3, 1.035e-2],
    ],
}


# ============================================
# INITIAL CONDITION
# ============================================

def create_initial_state():

    C = np.full(
        (nx, ny),
        C0,
        dtype=float
    )

    # Central low-oxygen wound region
    # Radius = 15 grid cells = 1.5 mm
    radius = 15

    y, x = np.ogrid[:nx, :ny]

    mask = (
        (x - centre_y) ** 2
        + (y - centre_x) ** 2
        <= radius ** 2
    )

    # Initial hypoxic wound-centre concentration
    C[mask] = 0.05

    # Fixed boundary conditions
    C[0, :] = C_boundary
    C[-1, :] = C_boundary
    C[:, 0] = C_boundary
    C[:, -1] = C_boundary

    return C


# ============================================
# MODEL EVALUATION FUNCTION
# ============================================

def evaluate_model(
    D_value,
    Vmax_value,
    Km_value,
    rho_value,
    beta_value
):

    C = create_initial_state()

    hypoxia_exit_time = None

    for step in range(n_steps):

        dCdt = full_oxygen_model(
            C,
            D_value,
            Vmax_value,
            Km_value,
            rho_value,
            beta_value,
            Cb,
            dx,
            dy,
        )

        # Forward Euler update
        C = C + dt_sobol * dCdt

        # Oxygen concentration cannot be negative
        C = np.maximum(
            C,
            0.0
        )

        # Reapply fixed boundary conditions
        C[0, :] = C_boundary
        C[-1, :] = C_boundary
        C[:, 0] = C_boundary
        C[:, -1] = C_boundary

        # Record the first time the wound centre
        # reaches or exceeds the hypoxia threshold
        if (
            hypoxia_exit_time is None
            and C[centre_x, centre_y] >= C_hyp
        ):

            hypoxia_exit_time = (
                (step + 1) * dt_sobol
            )

    # Final wound-centre oxygen concentration
    final_centre_oxygen = C[
        centre_x,
        centre_y
    ]

    # If the centre remains hypoxic for the
    # entire simulation, assign 600 s.
    if hypoxia_exit_time is None:

        hypoxia_exit_time = T_sobol

    return (
        final_centre_oxygen,
        hypoxia_exit_time
    )


# ============================================
# GENERATE SOBOL PARAMETER SAMPLES
# ============================================

print(
    "======================================"
)

print(
    "SOBOL SENSITIVITY ANALYSIS - N = 256"
)

print(
    "======================================"
)

print(
    "\nGenerating parameter samples..."
)


param_values = sobol_sample.sample(
    problem,
    N,
    calc_second_order=False,
    seed=42,
)


number_of_runs = len(
    param_values
)


print(
    "Number of parameters:",
    problem["num_vars"]
)

print(
    "Base sample size N:",
    N
)

print(
    "Total model evaluations:",
    number_of_runs
)

print(
    "Sobol time step:",
    dt_sobol,
    "seconds"
)

print(
    "Steps per simulation:",
    n_steps
)


# ============================================
# OUTPUT ARRAYS
# ============================================

Y_final_oxygen = np.zeros(
    number_of_runs
)

Y_exit_time = np.zeros(
    number_of_runs
)


# ============================================
# RUN ALL MODEL EVALUATIONS
# ============================================

print(
    "\nRunning Sobol model evaluations..."
)

start_time = time.perf_counter()


for i, params in enumerate(
    param_values
):

    (
        D_value,
        Vmax_value,
        Km_value,
        rho_value,
        beta_value,
    ) = params

    (
        final_oxygen,
        exit_time,
    ) = evaluate_model(
        D_value,
        Vmax_value,
        Km_value,
        rho_value,
        beta_value,
    )

    Y_final_oxygen[i] = (
        final_oxygen
    )

    Y_exit_time[i] = (
        exit_time
    )

    # Print progress every 50 simulations
    if (
        (i + 1) % 50 == 0
        or (i + 1) == number_of_runs
    ):

        elapsed = (
            time.perf_counter()
            - start_time
        )

        percent_complete = (
            (i + 1)
            / number_of_runs
            * 100
        )

        print(
            f"Completed {i + 1} / "
            f"{number_of_runs} runs "
            f"({percent_complete:.1f}%) "
            f"- {elapsed:.1f} s elapsed"
        )


end_time = time.perf_counter()

total_runtime = (
    end_time
    - start_time
)


# ============================================
# MODEL OUTPUT RANGE
# ============================================

print(
    "\n======================================"
)

print(
    "MODEL OUTPUT RANGE"
)

print(
    "======================================"
)


print(
    "Final centre oxygen range:",
    np.min(Y_final_oxygen),
    "to",
    np.max(Y_final_oxygen)
)


print(
    "Hypoxia exit time range:",
    np.min(Y_exit_time),
    "to",
    np.max(Y_exit_time),
    "seconds"
)


print(
    "Total runtime:",
    total_runtime,
    "seconds"
)


print(
    "Total runtime:",
    total_runtime / 60,
    "minutes"
)


# ============================================
# SOBOL ANALYSIS
# FINAL CENTRE OXYGEN
# ============================================

print(
    "\n======================================"
)

print(
    "SOBOL RESULTS - FINAL CENTRE OXYGEN"
)

print(
    "======================================"
)


Si_oxygen = sobol.analyze(
    problem,
    Y_final_oxygen,
    calc_second_order=False,
    print_to_console=False,
)


print(
    "\nParameter      S1          ST"
)

print(
    "--------------------------------"
)


for i, name in enumerate(
    problem["names"]
):

    print(
        f"{name:<12}"
        f"{Si_oxygen['S1'][i]:<12.4f}"
        f"{Si_oxygen['ST'][i]:<12.4f}"
    )


# ============================================
# SOBOL ANALYSIS
# TIME TO EXIT HYPOXIA
# ============================================

print(
    "\n======================================"
)

print(
    "SOBOL RESULTS - TIME TO EXIT HYPOXIA"
)

print(
    "======================================"
)


Si_exit = sobol.analyze(
    problem,
    Y_exit_time,
    calc_second_order=False,
    print_to_console=False,
)


print(
    "\nParameter      S1          ST"
)

print(
    "--------------------------------"
)


for i, name in enumerate(
    problem["names"]
):

    print(
        f"{name:<12}"
        f"{Si_exit['S1'][i]:<12.4f}"
        f"{Si_exit['ST'][i]:<12.4f}"
    )


# ============================================
# PRINT CONFIDENCE INTERVALS
# ============================================

print(
    "\n======================================"
)

print(
    "CONFIDENCE INTERVALS - FINAL OXYGEN"
)

print(
    "======================================"
)


print(
    "\nParameter      S1_conf     ST_conf"
)

print(
    "------------------------------------"
)


for i, name in enumerate(
    problem["names"]
):

    print(
        f"{name:<12}"
        f"{Si_oxygen['S1_conf'][i]:<12.4f}"
        f"{Si_oxygen['ST_conf'][i]:<12.4f}"
    )


print(
    "\n======================================"
)

print(
    "CONFIDENCE INTERVALS - EXIT TIME"
)

print(
    "======================================"
)


print(
    "\nParameter      S1_conf     ST_conf"
)

print(
    "------------------------------------"
)


for i, name in enumerate(
    problem["names"]
):

    print(
        f"{name:<12}"
        f"{Si_exit['S1_conf'][i]:<12.4f}"
        f"{Si_exit['ST_conf'][i]:<12.4f}"
    )


# ============================================
# SAVE RAW PARAMETER SAMPLES
# ============================================

np.savetxt(
    "sobol_parameter_samples_N256.csv",
    param_values,
    delimiter=",",
    header="D,Vmax,Km,rho_cell,beta",
    comments="",
)


# ============================================
# SAVE MODEL OUTPUTS
# ============================================

np.savetxt(
    "sobol_model_outputs_N256.csv",

    np.column_stack(
        (
            Y_final_oxygen,
            Y_exit_time,
        )
    ),

    delimiter=",",

    header=(
        "final_centre_oxygen,"
        "time_to_exit_hypoxia"
    ),

    comments="",
)


# ============================================
# SAVE SOBOL INDEX RESULTS
# ============================================

sobol_results_oxygen = np.column_stack(
    (
        Si_oxygen["S1"],
        Si_oxygen["S1_conf"],
        Si_oxygen["ST"],
        Si_oxygen["ST_conf"],
    )
)


np.savetxt(
    "sobol_indices_final_oxygen_N256.csv",

    sobol_results_oxygen,

    delimiter=",",

    header=(
        "S1,S1_conf,ST,ST_conf"
    ),

    comments="",
)


sobol_results_exit = np.column_stack(
    (
        Si_exit["S1"],
        Si_exit["S1_conf"],
        Si_exit["ST"],
        Si_exit["ST_conf"],
    )
)


np.savetxt(
    "sobol_indices_exit_time_N256.csv",

    sobol_results_exit,

    delimiter=",",

    header=(
        "S1,S1_conf,ST,ST_conf"
    ),

    comments="",
)


# ============================================
# FIGURE SETUP
# ============================================

names = problem["names"]

x = np.arange(
    len(names)
)

width = 0.35


# ============================================
# FIGURE 1
# FINAL CENTRE OXYGEN SOBOL INDICES
# ============================================

plt.figure(
    figsize=(8, 5)
)


plt.bar(
    x - width / 2,
    Si_oxygen["S1"],
    width,
    label="First-order S1"
)


plt.bar(
    x + width / 2,
    Si_oxygen["ST"],
    width,
    label="Total-order ST"
)


plt.xticks(
    x,
    names
)


plt.ylabel(
    "Sobol sensitivity index"
)


plt.xlabel(
    "Model parameter"
)


plt.title(
    "Sobol Sensitivity - Final Centre Oxygen (N=256)"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    "sobol_final_oxygen_N256.png",
    dpi=300
)


plt.show()


# ============================================
# FIGURE 2
# TIME TO EXIT HYPOXIA SOBOL INDICES
# ============================================

plt.figure(
    figsize=(8, 5)
)


plt.bar(
    x - width / 2,
    Si_exit["S1"],
    width,
    label="First-order S1"
)


plt.bar(
    x + width / 2,
    Si_exit["ST"],
    width,
    label="Total-order ST"
)


plt.xticks(
    x,
    names
)


plt.ylabel(
    "Sobol sensitivity index"
)


plt.xlabel(
    "Model parameter"
)


plt.title(
    "Sobol Sensitivity - Time to Exit Hypoxia (N=256)"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    "sobol_hypoxia_exit_time_N256.png",
    dpi=300
)


plt.show()


# ============================================
# FINISHED
# ============================================

print(
    "\n======================================"
)

print(
    "SOBOL ANALYSIS N = 256 COMPLETE"
)

print(
    "======================================"
)


print(
    "\nSaved files:"
)


print(
    "sobol_parameter_samples_N256.csv"
)

print(
    "sobol_model_outputs_N256.csv"
)

print(
    "sobol_indices_final_oxygen_N256.csv"
)

print(
    "sobol_indices_exit_time_N256.csv"
)

print(
    "sobol_final_oxygen_N256.png"
)

print(
    "sobol_hypoxia_exit_time_N256.png"
)