
import numpy as np


# ============================================
# OXYGENATED WOUND HEALING MODEL
# ============================================


# --------------------------------------------
# 1. ORIGINAL LAPLACIAN
# --------------------------------------------

def laplacian(C, dx, dy):
    """
    Original 2D finite-difference Laplacian.

    Calculates diffusion at interior grid points.
    Boundary conditions must be applied separately.
    """

    lap = np.zeros_like(C)

    lap[1:-1, 1:-1] = (
        (
            C[2:, 1:-1]
            - 2 * C[1:-1, 1:-1]
            + C[:-2, 1:-1]
        ) / dx**2
        +
        (
            C[1:-1, 2:]
            - 2 * C[1:-1, 1:-1]
            + C[1:-1, :-2]
        ) / dy**2
    )

    return lap


# --------------------------------------------
# 2. CONSERVATIVE ZERO-FLUX LAPLACIAN
# --------------------------------------------

def zero_flux_laplacian(C, dx, dy):
    """
    Conservative 2D diffusion discretisation
    with zero diffusive flux at outer boundaries.

    Internal face flux contributions are equal
    and opposite, conserving the discrete sum
    of oxygen during diffusion-only simulations.

    This matches the formulation verified in
    test_zero_flux.py.
    """

    lap = np.zeros_like(C)

    differences_x = (
        C[1:, :] - C[:-1, :]
    ) / dx**2

    lap[:-1, :] += differences_x
    lap[1:, :] -= differences_x

    differences_y = (
        C[:, 1:] - C[:, :-1]
    ) / dy**2

    lap[:, :-1] += differences_y
    lap[:, 1:] -= differences_y

    return lap


# --------------------------------------------
# 3. OXYGEN DIFFUSION
# --------------------------------------------

def diffusion(
    C,
    D,
    dx,
    dy,
    boundary_type="original"
):
    """
    Oxygen diffusion term:

        D * Laplacian(C)

    boundary_type:
        "original"  - original interior Laplacian
        "zero_flux" - conservative zero-flux
    """

    if boundary_type == "original":
        lap = laplacian(C, dx, dy)

    elif boundary_type == "zero_flux":
        lap = zero_flux_laplacian(C, dx, dy)

    else:
        raise ValueError(
            "Unknown boundary type: "
            + str(boundary_type)
        )

    return D * lap


# --------------------------------------------
# 4. CELLULAR OXYGEN CONSUMPTION
# --------------------------------------------

def oxygen_consumption(C, Vmax, Km, rho_cell):
    """
    Michaelis-Menten oxygen consumption.

        M(C) = rho_cell * Vmax * C / (Km + C)
    """

    consumption = (
        rho_cell
        * Vmax
        * C
        / (Km + C)
    )

    return consumption


# --------------------------------------------
# 5. MICROVASCULAR PERFUSION
# --------------------------------------------

def perfusion(C, beta, Cb):
    """
    Oxygen supply from perfusion.

        P(C) = beta * (Cb - C)

    beta may be a scalar or a 2D field.
    """

    return beta * (Cb - C)


# --------------------------------------------
# 6. DIFFUSION + CONSUMPTION MODEL
# --------------------------------------------

def diffusion_consumption_model(
    C,
    D,
    Vmax,
    Km,
    rho_cell,
    dx,
    dy,
    boundary_type="original"
):
    """
    Combined diffusion and consumption.

        dC/dt = D * Laplacian(C) - M(C)
    """

    diffusion_term = diffusion(
        C,
        D,
        dx,
        dy,
        boundary_type=boundary_type
    )

    consumption_term = oxygen_consumption(
        C,
        Vmax,
        Km,
        rho_cell
    )

    return diffusion_term - consumption_term


# --------------------------------------------
# 7. FULL OXYGEN TRANSPORT MODEL
# --------------------------------------------

def full_oxygen_model(
    C,
    D,
    Vmax,
    Km,
    rho_cell,
    beta,
    Cb,
    dx,
    dy,
    boundary_type="original"
):
    """
    Full oxygen transport model.

        dC/dt =
            D * Laplacian(C)
            - rho_cell * Vmax * C / (Km + C)
            + beta * (Cb - C)

    For the revised wound model, use:

        boundary_type="zero_flux"

    Do not overwrite boundary values when
    using the conservative zero-flux method.
    """

    diffusion_term = diffusion(
        C,
        D,
        dx,
        dy,
        boundary_type=boundary_type
    )

    consumption_term = oxygen_consumption(
        C,
        Vmax,
        Km,
        rho_cell
    )

    perfusion_term = perfusion(
        C,
        beta,
        Cb
    )

    return (
        diffusion_term
        - consumption_term
        + perfusion_term
    )


# --------------------------------------------
# 8. WOUND-BED GEOMETRY
# --------------------------------------------

def create_wound_mask(Nx, Ny, radius=15):
    """
    Create a circular wound-bed region.

    True  = wound-bed tissue
    False = surrounding tissue

    Radius is measured in grid cells.
    """

    x, y = np.ogrid[:Nx, :Ny]

    centre_x = Nx // 2
    centre_y = Ny // 2

    distance_squared = (
        (x - centre_x)**2
        + (y - centre_y)**2
    )

    return distance_squared < radius**2


# --------------------------------------------
# 9. SPATIALLY VARYING PERFUSION
# --------------------------------------------

def create_perfusion_field(
    wound_mask,
    beta_surrounding,
    wound_perfusion_fraction=0.5
):
    """
    Create a spatial perfusion coefficient field.

    Outside wound:
        beta = beta_surrounding

    Inside wound:
        beta = beta_surrounding
               * wound_perfusion_fraction

    The default 0.5 is a hypothetical test
    assumption, not a validated physiological
    wound-perfusion ratio.
    """

    if not 0 <= wound_perfusion_fraction <= 1:
        raise ValueError(
            "wound_perfusion_fraction must "
            "be between 0 and 1."
        )

    if beta_surrounding < 0:
        raise ValueError(
            "beta_surrounding must be non-negative."
        )

    beta_field = np.full(
        wound_mask.shape,
        beta_surrounding,
        dtype=float
    )

    beta_field[wound_mask] = (
        beta_surrounding
        * wound_perfusion_fraction
    )

    return beta_field
