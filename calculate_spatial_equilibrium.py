
"""
Calculate model-derived spatial oxygen equilibria.

Uses the existing diffusion-consumption-perfusion equation
and the corrected scenario-specific perfusion fields.

These are exploratory mathematical equilibrium solutions,
NOT clinically validated initial wound conditions.

Does not modify existing simulation files or results.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import final_gradient_simulation_corrected as f
from model import full_oxygen_model


OUTPUT_DIR = Path("spatial_equilibrium_results")

DT = 0.1
MAX_TIME = 7200.0
CHECK_INTERVAL = 10.0

# Stop when the maximum absolute rate across the domain
# is below this value.
RATE_TOLERANCE = 1e-5  # mmHg/s

SOURCE = f.SOURCE_CONCENTRATION


def local_equilibrium(beta):
    """
    Solve local supply = consumption without diffusion,
    using bisection.
    """
    low = 0.0
    high = SOURCE

    for _ in range(100):
        middle = (low + high) / 2.0

        supply = beta * (SOURCE - middle)

        consumption = (
            f.rho_cell
            * f.Vmax
            * middle
            / (f.Km + middle)
        )

        if supply > consumption:
            low = middle
        else:
            high = middle

    return 0.5 * (low + high)


def make_perfusion_field(wound_reference_beta):
    """
    Shared healthy surrounding perfusion.
    Different perfusion inside the wound.
    """
    beta = np.full(
        (f.NX, f.NY),
        f.beta_normal,
        dtype=float,
    )

    beta[f.WOUND_MASK] = (
        wound_reference_beta
        * f.WOUND_PERFUSION_FRACTION
    )

    return beta


def derivative(C, beta):
    return full_oxygen_model(
        C,
        f.D,
        f.Vmax,
        f.Km,
        f.rho_cell,
        beta,
        SOURCE,
        f.dx,
        f.dy,
        boundary_type="zero_flux",
    )


def calculate_equilibrium(name, wound_reference_beta):
    beta = make_perfusion_field(wound_reference_beta)

    # Numerical starting guess for relaxation.
    # This is not a proposed physiological initial state.
    C = np.full(
        (f.NX, f.NY),
        local_equilibrium(f.beta_normal),
        dtype=float,
    )

    wound_beta = (
        wound_reference_beta
        * f.WOUND_PERFUSION_FRACTION
    )

    C[f.WOUND_MASK] = local_equilibrium(wound_beta)

    max_steps = int(round(MAX_TIME / DT))
    check_every = int(round(CHECK_INTERVAL / DT))

    converged = False
    final_rate = None
    final_time = 0.0

    for step in range(max_steps + 1):
        if step % check_every == 0:
            rate = derivative(C, beta) / f.alpha_O2
            final_rate = float(np.max(np.abs(rate)))
            final_time = step * DT

            if final_rate < RATE_TOLERANCE:
                converged = True
                break

        if step == max_steps:
            break

        C = C + DT * derivative(C, beta)

        if not np.all(np.isfinite(C)) or np.any(C < 0):
            raise ValueError(
                f"Invalid concentration in {name} simulation."
            )

    po2 = C / f.alpha_O2

    print(f"\n{name.upper()} EQUILIBRIUM")
    print(f"Converged: {converged}")
    print(f"Elapsed model time: {final_time:.1f} s")
    print(
        "Maximum absolute oxygen rate:",
        f"{final_rate:.8f} mmHg/s"
    )
    print(
        "Wound-centre oxygen:",
        f"{po2[f.NX // 2, f.NY // 2]:.4f} mmHg"
    )
    print(
        "Mean wound oxygen:",
        f"{np.mean(po2[f.WOUND_MASK]):.4f} mmHg"
    )
    print(
        "Mean surrounding oxygen:",
        f"{np.mean(po2[f.SURROUNDING_MASK]):.4f} mmHg"
    )

    return {
        "name": name,
        "field": po2,
        "converged": converged,
        "time_s": final_time,
        "max_rate": final_rate,
    }


def save_results(normal, diabetic):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save complete spatial fields for later analysis.
    np.save(
        OUTPUT_DIR / "normal_equilibrium_po2.npy",
        normal["field"],
    )
    np.save(
        OUTPUT_DIR / "diabetic_equilibrium_po2.npy",
        diabetic["field"],
    )

    results = []

    for result in (normal, diabetic):
        field = result["field"]

        results.append({
            "condition": result["name"],
            "converged": result["converged"],
            "time_s": result["time_s"],
            "max_rate_mmHg_per_s": result["max_rate"],
            "centre_po2_mmHg": field[
                f.NX // 2, f.NY // 2
            ],
            "mean_wound_po2_mmHg": np.mean(
                field[f.WOUND_MASK]
            ),
            "mean_surrounding_po2_mmHg": np.mean(
                field[f.SURROUNDING_MASK]
            ),
        })

    with (
        OUTPUT_DIR / "equilibrium_summary.csv"
    ).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    fields = [
        normal["field"],
        diabetic["field"],
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    vmin = min(float(np.min(x)) for x in fields)
    vmax = max(float(np.max(x)) for x in fields)

    for ax, field, title in zip(
        axes, fields, ["Normal", "Diabetic"]
    ):
        image = ax.imshow(
            field.T,
            origin="lower",
            extent=[
                f.X_MM[0], f.X_MM[-1],
                f.Y_MM[0], f.Y_MM[-1],
            ],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        ax.contour(
            f.X_MM,
            f.Y_MM,
            f.WOUND_MASK.T.astype(float),
            levels=[0.5],
            colors="white",
        )

        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.colorbar(
        image,
        ax=axes,
        label="Oxygen partial pressure (mmHg)",
    )

    fig.suptitle(
        "Model-derived 2D spatial oxygen equilibria"
    )

    fig.savefig(
        OUTPUT_DIR / "spatial_equilibrium_maps.png",
        dpi=250,
    )
    plt.close(fig)

    # Compare oxygen across the wound centreline.
    fig, ax = plt.subplots(figsize=(9, 5))

    centre_y = f.NY // 2

    ax.plot(
        f.X_MM,
        normal["field"][:, centre_y],
        label="Normal equilibrium",
        linewidth=2,
    )

    ax.plot(
        f.X_MM,
        diabetic["field"][:, centre_y],
        label="Diabetic equilibrium",
        linewidth=2,
    )

    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Oxygen partial pressure (mmHg)")
    ax.set_title("Spatial equilibrium across wound centre")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "equilibrium_centreline.png",
        dpi=250,
    )
    plt.close(fig)


def main():
    if (
        f.D * DT
        * (1.0 / f.dx**2 + 1.0 / f.dy**2)
        > 0.5
    ):
        raise ValueError(
            "Time step exceeds diffusion stability limit."
        )

    print("=" * 60)
    print("SPATIAL OXYGEN EQUILIBRIUM ANALYSIS")
    print("=" * 60)

    normal = calculate_equilibrium(
        "normal",
        f.beta_normal,
    )

    diabetic = calculate_equilibrium(
        "diabetic",
        f.beta_diabetic,
    )

    if not (normal["converged"] and diabetic["converged"]):
        print(
            "\nWARNING: At least one scenario did not "
            "meet the equilibrium tolerance."
        )

    save_results(normal, diabetic)

    print("\nResults saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
