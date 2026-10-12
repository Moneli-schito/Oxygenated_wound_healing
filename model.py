import numpy as np


# ============================================
# OXYGENATED WOUND HEALING MODEL
# Model Equations
# ============================================


def laplacian(C, dx, dy):
    """
    Calculate the 2D Laplacian of oxygen concentration
    using the Finite Difference Method.

    The Laplacian represents how oxygen concentration
    varies spatially around each grid point.
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


def diffusion(C, D, dx, dy):
    """
    Oxygen diffusion term:

        D * ∇²C

    This represents oxygen spreading from regions
    of higher concentration to regions of lower
    concentration.
    """

    diffusion_term = D * laplacian(
        C,
        dx,
        dy
    )

    return diffusion_term


def oxygen_consumption(C, Vmax, Km, rho_cell):
    """
    Cellular oxygen consumption using
    Michaelis-Menten kinetics.

        M(C) =
        rho_cell * Vmax * C / (Km + C)

    This represents oxygen removed from the
    tissue by cellular metabolism.
    """

    consumption = (
        rho_cell
        * Vmax
        * C
        / (Km + C)
    )

    return consumption


def perfusion(C, beta, Cb):
    """
    Oxygen supply from microvascular perfusion.

        P(C) = beta * (Cb - C)

    where:

        beta = effective perfusion coefficient (1/s)
        Cb   = blood/source oxygen concentration
        C    = local tissue oxygen concentration

    If tissue oxygen is lower than blood oxygen,
    this term supplies oxygen to the tissue.
    """

    perfusion_term = beta * (Cb - C)

    return perfusion_term


def diffusion_consumption_model(
    C,
    D,
    Vmax,
    Km,
    rho_cell,
    dx,
    dy
):
    """
    Combined diffusion + oxygen consumption model.

        dC/dt = D∇²C - M(C)

    Diffusion redistributes oxygen through the
    tissue while cellular consumption removes it.
    """

    diffusion_term = diffusion(
        C,
        D,
        dx,
        dy
    )

    consumption_term = oxygen_consumption(
        C,
        Vmax,
        Km,
        rho_cell
    )

    dCdt = (
        diffusion_term
        - consumption_term
    )

    return dCdt


def full_oxygen_model(
    C,
    D,
    Vmax,
    Km,
    rho_cell,
    beta,
    Cb,
    dx,
    dy
):
    """
    Complete oxygen transport model.

    The model combines:

        1. Oxygen diffusion
        2. Cellular oxygen consumption
        3. Microvascular perfusion

    Governing equation:

        dC/dt =
            D∇²C
            - rho_cell * Vmax * C / (Km + C)
            + beta * (Cb - C)

    or:

        Change in oxygen
        =
        Diffusion
        - Consumption
        + Perfusion
    """

    # ----------------------------------------
    # 1. Oxygen diffusion
    # ----------------------------------------

    diffusion_term = diffusion(
        C,
        D,
        dx,
        dy
    )

    # ----------------------------------------
    # 2. Cellular oxygen consumption
    # ----------------------------------------

    consumption_term = oxygen_consumption(
        C,
        Vmax,
        Km,
        rho_cell
    )

    # ----------------------------------------
    # 3. Oxygen supplied by perfusion
    # ----------------------------------------

    perfusion_term = perfusion(
        C,
        beta,
        Cb
    )

    # ----------------------------------------
    # Complete coupled model
    # ----------------------------------------

    dCdt = (
        diffusion_term
        - consumption_term
        + perfusion_term
    )

    return dCdt