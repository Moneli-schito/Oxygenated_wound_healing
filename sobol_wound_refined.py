
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol

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
    alpha_O2,
)

from model import (
    full_oxygen_model,
    create_wound_mask,
    create_perfusion_field,
)


# ============================================
# 1. ANALYSIS SETTINGS
# ============================================

N = 256
DT = 0.1
T = 600.0

SEED = 42
WOUND_RADIUS_CELLS = 15
INITIAL_WOUND_CONCENTRATION = 0.05

OUTPUT_DIRECTORY = "sobol_refined_results"

# The timestep was checked at the normal and
# diabetic baseline parameter values.
# This does not establish convergence for every
# parameter combination in the Sobol sample.


# ============================================
# 2. UNCERTAIN PARAMETERS
# ============================================

# These ranges are exploratory assumptions.
#
# D, Vmax, Km and rho_cell: +/-20%
# Surrounding beta: 50% to 150% of normal
# Wound perfusion fraction: 0.25 to 0.75
#
# All parameters are sampled independently.
# Sensitivity rankings depend on these ranges.

problem = {
    "num_vars": 6,
    "names": [
        "D",
        "Vmax",
        "Km",
        "rho_cell",
        "beta_surrounding",
        "wound_perfusion_fraction",
    ],
    "bounds": [
        [0.8 * D, 1.2 * D],
        [0.8 * Vmax, 1.2 * Vmax],
        [0.8 * Km, 1.2 * Km],
        [0.8 * rho_cell, 1.2 * rho_cell],
        [0.5 * beta_normal, 1.5 * beta_normal],
        [0.25, 0.75],
    ],
}


# ============================================
# 3. SPATIAL GRID
# ============================================

NX = int(round(Lx / dx)) + 1
NY = int(round(Ly / dy)) + 1

CENTRE_X = NX // 2
CENTRE_Y = NY // 2

WOUND_MASK = create_wound_mask(
    NX,
    NY,
    radius=WOUND_RADIUS_CELLS,
)

SURROUNDING_MASK = ~WOUND_MASK

N_STEPS = int(round(T / DT))


def initial_state():
    C = np.full(
        (NX, NY),
        C0,
        dtype=float,
    )

    C[WOUND_MASK] = INITIAL_WOUND_CONCENTRATION

    return C


# ============================================
# 4. MODEL EVALUATION
# ============================================

def evaluate_model(parameter_values):
    (
        D_value,
        Vmax_value,
        Km_value,
        rho_value,
        beta_surrounding,
        wound_fraction,
    ) = parameter_values

    beta_field = create_perfusion_field(
        WOUND_MASK,
        beta_surrounding,
        wound_perfusion_fraction=wound_fraction,
    )

    C = initial_state()

    exit_time = None

    # Integral of hypoxic wound fraction over time.
    # Units: seconds, because the fraction is 0 to 1.
    #
    # For example, 100% hypoxia lasting 10 seconds
    # gives 10 hypoxic-fraction-seconds.
    cumulative_hypoxic_exposure = 0.0

    # Initial wound hypoxic fraction
    previous_hypoxic_fraction = np.mean(
        C[WOUND_MASK] < C_hyp
    )

    for step in range(N_STEPS):
        previous_centre = C[CENTRE_X, CENTRE_Y]

        dCdt = full_oxygen_model(
            C,
            D_value,
            Vmax_value,
            Km_value,
            rho_value,
            beta_field,
            Cb,
            dx,
            dy,
            boundary_type="zero_flux",
        )

        C = C + DT * dCdt

        if not np.all(np.isfinite(C)):
            raise RuntimeError(
                "Non-finite oxygen concentration "
                f"at timestep {step + 1}."
            )

        if np.min(C) < -1e-10:
            raise RuntimeError(
                "Negative oxygen concentration "
                f"at timestep {step + 1}."
            )

        C = np.maximum(C, 0.0)

        current_centre = C[CENTRE_X, CENTRE_Y]

        # Linear interpolation gives a more precise
        # estimate than rounding to the timestep.
        if (
            exit_time is None
            and previous_centre < C_hyp
            and current_centre >= C_hyp
        ):
            change = current_centre - previous_centre

            fraction = (
                (C_hyp - previous_centre) / change
            )

            exit_time = (
                step + fraction
            ) * DT

        current_hypoxic_fraction = np.mean(
            C[WOUND_MASK] < C_hyp
        )

        # Trapezoidal time integration
        cumulative_hypoxic_exposure += (
            0.5
            * (
                previous_hypoxic_fraction
                + current_hypoxic_fraction
            )
            * DT
        )

        previous_hypoxic_fraction = (
            current_hypoxic_fraction
        )

    final_centre_pO2 = (
        C[CENTRE_X, CENTRE_Y] / alpha_O2
    )

    final_mean_wound_pO2 = (
        np.mean(C[WOUND_MASK]) / alpha_O2
    )

    final_mean_surrounding_pO2 = (
        np.mean(C[SURROUNDING_MASK]) / alpha_O2
    )

    final_wound_surrounding_difference = (
        final_mean_surrounding_pO2
        - final_mean_wound_pO2
    )

    final_hypoxic_fraction = np.mean(
        C[WOUND_MASK] < C_hyp
    )

    # Retain the distinction between a threshold
    # crossing and no crossing within 600 seconds.
    #
    # For the Sobol output, a non-crossing case
    # receives T as a right-censored value.
    # Such results must be interpreted as
    # "time-to-threshold capped at 600 seconds",
    # not an actual threshold-crossing time.
    crossed_threshold = exit_time is not None

    capped_exit_time = (
        exit_time
        if crossed_threshold
        else T
    )

    return {
        "centre_pO2": final_centre_pO2,
        "mean_wound_pO2": final_mean_wound_pO2,
        "wound_surrounding_difference":
            final_wound_surrounding_difference,
        "capped_exit_time": capped_exit_time,
        "crossed_threshold": int(
            crossed_threshold
        ),
        "cumulative_hypoxic_exposure":
            cumulative_hypoxic_exposure,
        "final_hypoxic_fraction":
            final_hypoxic_fraction,
    }


# ============================================
# 5. SAVE UTILITIES
# ============================================

def save_csv(filename, data, header):
    path = os.path.join(
        OUTPUT_DIRECTORY,
        filename,
    )

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=header,
        comments="",
    )


def save_sobol_plot(name, indices):
    names = problem["names"]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(11, 5.5))

    plt.bar(
        x - width / 2,
        indices["S1"],
        width,
        label="First-order S1",
    )

    plt.bar(
        x + width / 2,
        indices["ST"],
        width,
        label="Total-order ST",
    )

    plt.xticks(
        x,
        names,
        rotation=25,
        ha="right",
    )

    plt.ylabel("Sobol sensitivity index")
    plt.xlabel("Model parameter")

    plt.title(
        f"Refined Wound Model: {name} (N={N})"
    )

    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    filename = (
        f"sobol_{safe_name}_N{N}.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIRECTORY,
            filename,
        ),
        dpi=300,
    )

    plt.close()


# ============================================
# 6. MAIN ANALYSIS
# ============================================

def main():
    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    print("=" * 65)
    print("REFINED WOUND MODEL - SOBOL ANALYSIS")
    print("=" * 65)

    print(f"Branch study: revised wound model")
    print(f"Grid: {NX} x {NY}")
    print(f"Duration: {T} s")
    print(f"Timestep: {DT} s")
    print(f"Base sample size: {N}")
    print("Boundary: conservative zero flux")
    print("Parameter ranges: exploratory")
    print(
        "pO2 conversion: provisional plasma "
        "solubility approximation"
    )

    print("\nGenerating Sobol samples...")

    samples = sobol_sample.sample(
        problem,
        N,
        calc_second_order=False,
        seed=SEED,
    )

    number_of_runs = len(samples)

    print(
        f"Total model evaluations: "
        f"{number_of_runs}"
    )

    save_csv(
        f"parameter_samples_N{N}.csv",
        samples,
        ",".join(problem["names"]),
    )

    output_names = [
        "centre_pO2",
        "mean_wound_pO2",
        "wound_surrounding_difference",
        "capped_exit_time",
        "cumulative_hypoxic_exposure",
    ]

    all_output_names = (
        output_names
        + [
            "crossed_threshold",
            "final_hypoxic_fraction",
        ]
    )

    outputs = {
        name: np.zeros(number_of_runs)
        for name in all_output_names
    }

    start_time = time.perf_counter()

    print("\nRunning model evaluations...")

    for i, sample in enumerate(samples):
        result = evaluate_model(sample)

        for name in all_output_names:
            outputs[name][i] = result[name]

        if (
            (i + 1) % 25 == 0
            or i + 1 == number_of_runs
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Completed {i + 1}/"
                f"{number_of_runs} "
                f"({100 * (i + 1) / number_of_runs:.1f}%) "
                f"- {elapsed / 60:.2f} minutes",
                flush=True,
            )

    runtime = time.perf_counter() - start_time

    output_matrix = np.column_stack(
        [
            outputs[name]
            for name in all_output_names
        ]
    )

    save_csv(
        f"model_outputs_N{N}.csv",
        output_matrix,
        ",".join(all_output_names),
    )

    print("\n" + "=" * 65)
    print("MODEL OUTPUT SUMMARY")
    print("=" * 65)

    for name in all_output_names:
        values = outputs[name]

        print(
            f"{name}: "
            f"{np.min(values):.6f} to "
            f"{np.max(values):.6f}"
        )

    non_crossing_count = int(
        np.sum(
            outputs["crossed_threshold"] == 0
        )
    )

    print(
        f"\nNon-crossing simulations: "
        f"{non_crossing_count}/"
        f"{number_of_runs}"
    )

    print(
        f"Total simulation runtime: "
        f"{runtime / 60:.2f} minutes"
    )

    print("\n" + "=" * 65)
    print("SOBOL SENSITIVITY INDICES")
    print("=" * 65)

    for output_name in output_names:
        values = outputs[output_name]

        if not np.all(np.isfinite(values)):
            print(
                f"\nSkipping {output_name}: "
                "non-finite outputs."
            )
            continue

        if np.var(values) <= 1e-20:
            print(
                f"\nSkipping {output_name}: "
                "output variance is too small."
            )
            continue

        indices = sobol.analyze(
            problem,
            values,
            calc_second_order=False,
            print_to_console=False,
            seed=SEED,
        )

        print(f"\nOUTPUT: {output_name}")

        print(
            f"{'Parameter':<28}"
            f"{'S1':>10}"
            f"{'ST':>10}"
            f"{'S1 conf':>12}"
            f"{'ST conf':>12}"
        )

        for j, parameter_name in enumerate(
            problem["names"]
        ):
            print(
                f"{parameter_name:<28}"
                f"{indices['S1'][j]:>10.4f}"
                f"{indices['ST'][j]:>10.4f}"
                f"{indices['S1_conf'][j]:>12.4f}"
                f"{indices['ST_conf'][j]:>12.4f}"
            )

        index_matrix = np.column_stack(
            [
                indices["S1"],
                indices["S1_conf"],
                indices["ST"],
                indices["ST_conf"],
            ]
        )

        safe_name = output_name.lower()

        save_csv(
            f"indices_{safe_name}_N{N}.csv",
            index_matrix,
            "S1,S1_conf,ST,ST_conf",
        )

        save_sobol_plot(
            output_name,
            indices,
        )

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)

    print(
        f"Results saved in: "
        f"{OUTPUT_DIRECTORY}"
    )

    print(
        "\nInterpretation cautions:"
        "\n- Parameter ranges are exploratory."
        "\n- Sobol indices depend on those ranges."
        "\n- Wound perfusion fraction is hypothetical."
        "\n- Hypoxia threshold is provisional."
        "\n- pO2 conversion is not tissue-calibrated."
        "\n- Threshold crossing does not prove healing."
    )


if __name__ == "__main__":
    main()
