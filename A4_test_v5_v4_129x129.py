import numpy as np
import sys

# np.set_printoptions(linewidth=np.inf)
# np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

n_x = 129
n_y = 129

dx = 1.0 / n_x
dy = 1.0 / n_y

Re = 100


# ---------------------------------------------------------------------------
# Momentum link coefficients (A_p, A_e, A_w, A_n, A_s and sources)
# ---------------------------------------------------------------------------

def momentum_link_coefficients(u_star, u_face, v_face, p, source_x, source_y,
                               A_p, A_e, A_w, A_n, A_s):
    """
    Build momentum equation coefficients and source terms.
    Interior cells are vectorized; boundaries/corners kept explicit.
    """

    D_e = dy / (dx * Re)
    D_w = dy / (dx * Re)
    D_n = dx / (dy * Re)
    D_s = dx / (dy * Re)

    # -------------------------------
    # Interior cells: i=2..n_y-1, j=2..n_x-1 (vectorized)
    # -------------------------------
    i_int = slice(2, n_y)
    j_int = slice(2, n_x)

    # Fluxes
    F_e = dy * u_face[i_int, j_int]
    F_w = dy * u_face[i_int, slice(1, n_x - 1)]
    F_n = dx * v_face[slice(1, n_y - 1), j_int]
    F_s = dx * v_face[i_int, j_int]

    A_e[i_int, j_int] = D_e + np.maximum(0.0, -F_e)
    A_w[i_int, j_int] = D_w + np.maximum(0.0, F_w)
    A_n[i_int, j_int] = D_n + np.maximum(0.0, -F_n)
    A_s[i_int, j_int] = D_s + np.maximum(0.0, F_s)

    A_p[i_int, j_int] = (
            A_w[i_int, j_int] + A_e[i_int, j_int] +
            A_n[i_int, j_int] + A_s[i_int, j_int] +
            (F_e - F_w) + (F_n - F_s)
    )

    # pressure source terms (interior)
    p_w = p[i_int, slice(1, n_x - 1)]
    p_e = p[i_int, slice(3, n_x + 1)]
    p_s = p[slice(1, n_y - 1), j_int]
    p_n = p[slice(3, n_y + 1), j_int]

    source_x[i_int, j_int] = 0.5 * (p_w - p_e) * dx
    source_y[i_int, j_int] = 0.5 * (p_n - p_s) * dy

    # -------------------------------
    # Boundaries (same logic as original; scalar loops)
    # -------------------------------

    # left wall (j = 1)
    j = 1
    for i in range(2, n_y):
        F_e = dy * u_face[i, j]
        F_w = dy * u_face[i, j - 1]  # left face velocity is initialised to zero
        F_n = dx * v_face[i - 1, j]
        F_s = dx * v_face[i, j]

        A_e[i, j] = D_e + max(0.0, -F_e)
        A_w[i, j] = 2.0 * D_w + max(0.0, F_w)
        A_n[i, j] = D_n + max(0.0, -F_n)
        A_s[i, j] = D_s + max(0.0, F_s)
        A_p[i, j] = (
                A_w[i, j] + A_e[i, j] + A_n[i, j] + A_s[i, j] +
                (F_e - F_w) + (F_n - F_s)
        )

        source_x[i, j] = 0.5 * (p[i, j] - p[i, j + 1]) * dx  # P_o - 0.5(P_o+P_e)
        source_y[i, j] = 0.5 * (p[i + 1, j] - p[i - 1, j]) * dy

    # bottom wall (i = n_y)
    i = n_y
    for j in range(2, n_x):
        F_e = dy * u_face[i, j]
        F_w = dy * u_face[i, j - 1]
        F_n = dx * v_face[i - 1, j]
        F_s = dx * v_face[i, j]  # bottom wall v-velocity is zero

        A_e[i, j] = D_e + max(0.0, -F_e)
        A_w[i, j] = D_w + max(0.0, F_w)
        A_n[i, j] = D_n + max(0.0, -F_n)
        A_s[i, j] = 2.0 * D_s + max(0.0, F_s)
        A_p[i, j] = (
                A_w[i, j] + A_e[i, j] + A_n[i, j] + A_s[i, j] +
                (F_e - F_w) + (F_n - F_s)
        )

        source_x[i, j] = 0.5 * (p[i, j - 1] - p[i, j + 1]) * dx
        source_y[i, j] = 0.5 * (p[i, j] - p[i - 1, j]) * dy  # P_o - 0.5(P_o+P_n)

    # right wall (j = n_x)
    j = n_x
    for i in range(2, n_y):
        F_e = dy * u_face[i, j]
        F_w = dy * u_face[i, j - 1]  # right face velocity is initialised to zero
        F_n = dx * v_face[i - 1, j]
        F_s = dx * v_face[i, j]

        A_e[i, j] = D_e + max(0.0, -F_e)
        A_w[i, j] = 2.0 * D_w + max(0.0, F_w)
        A_n[i, j] = D_n + max(0.0, -F_n)
        A_s[i, j] = D_s + max(0.0, F_s)
        A_p[i, j] = (
                A_w[i, j] + A_e[i, j] + A_n[i, j] + A_s[i, j] +
                (F_e - F_w) + (F_n - F_s)
        )

        source_x[i, j] = 0.5 * (p[i, j - 1] - p[i, j]) * dx  # 0.5(P_w+P_o)-P_o
        source_y[i, j] = 0.5 * (p[i + 1, j] - p[i - 1, j]) * dy

    # top wall (i = 1)
    i = 1
    for j in range(2, n_x):
        F_e = dy * u_face[i, j]
        F_w = dy * u_face[i, j - 1]
        F_n = dx * v_face[i - 1, j]
        F_s = dx * v_face[i, j]

        A_e[i, j] = D_e + max(0.0, -F_e)
        A_w[i, j] = D_w + max(0.0, F_w)
        A_n[i, j] = 2.0 * D_n + max(0.0, -F_n)
        A_s[i, j] = D_s + max(0.0, F_s)
        A_p[i, j] = (
                A_w[i, j] + A_e[i, j] + A_n[i, j] + A_s[i, j] +
                (F_e - F_w) + (F_n - F_s)
        )

        source_x[i, j] = 0.5 * (p[i, j - 1] - p[i, j + 1]) * dx
        source_y[i, j] = 0.5 * (p[i + 1, j] - p[i, j]) * dy  # 0.5(P_s+P_o) - P_o

    # top left corner
    i = 1
    j = 1
    F_e = dy * u_face[i, j]
    F_w = dy * u_face[i, j - 1]
    F_n = dx * v_face[i - 1, j]
    F_s = dx * v_face[i, j]

    A_e[i, j] = D_e + max(0.0, -F_e)
    A_w[i, j] = 2.0 * D_w + max(0.0, F_w)
    A_n[i, j] = 2.0 * D_n + max(0.0, -F_n)
    A_s[i, j] = D_s + max(0.0, F_s)
    A_p[i, j] = (
            A_w[i, j] + A_e[i, j] + A_n[i, j] + A_s[i, j] +
            (F_e - F_w) + (F_n - F_s)
    )

    source_x[i, j] = 0.5 * (p[i, j] - p[i, j + 1]) * dx  # P_o - 0.5(P_o+P_e)
    source_y[i, j] = 0.5 * (p[i + 1, j] - p[i, j]) * dy  # 0.5(P_s+P_o) - P_o

    # top right corner
    i = 1
    j = n_x
    F_e = dy * u_face[i, j]
    F_w = dy * u_face[i, j - 1]
    F_n = dx * v_face[i - 1, j]
    F_s = dx * v_face[i, j]

    A_e[i, j] = D_e + max(0.0, -F_e)
    A_w[i, j] = 2.0 * D_w + max(0.0, F_w)
    A_n[i, j] = 2.0 * D_n + max(0.0, -F_n)
    A_s[i, j] = D_s + max(0.0, F_s)
    A_p[i, j] = (
            A_w[i, j] + A_e[i, j] + A_n[i, j] + A_s[i, j] +
            (F_e - F_w) + (F_n - F_s)
    )

    source_x[i, j] = 0.5 * (p[i, j - 1] - p[i, j]) * dx  # 0.5(P_w+P_o)-P_o
    source_y[i, j] = 0.5 * (p[i + 1, j] - p[i, j]) * dy  # 0.5(P_s+P_o) - P_o

    # bottom left corner
    i = n_y
    j = 1
    F_e = dy * u_face[i, j]
    F_w = dy * u_face[i, j - 1]
    F_n = dx * v_face[i - 1, j]
    F_s = dx * v_face[i, j]

    A_e[i, j] = D_e + max(0.0, -F_e)
    A_w[i, j] = 2.0 * D_w + max(0.0, F_w)
    A_n[i, j] = D_n + max(0.0, -F_n)
    A_s[i, j] = 2.0 * D_s + max(0.0, F_s)
    A_p[i, j] = (
            A_w[i, j] + A_e[i, j] + A_n[i, j] + A_s[i, j] +
            (F_e - F_w) + (F_n - F_s)
    )

    source_x[i, j] = 0.5 * (p[i, j] - p[i, j + 1]) * dx  # P_o - 0.5(P_o+P_e)
    source_y[i, j] = 0.5 * (p[i, j] - p[i - 1, j]) * dy  # P_o - 0.5(P_o+P_n)

    # bottom right corner
    i = n_y
    j = n_x
    F_e = dy * u_face[i, j]
    F_w = dy * u_face[i, j - 1]
    F_n = dx * v_face[i - 1, j]
    F_s = dx * v_face[i, j]

    A_e[i, j] = 2.0 * D_e + max(0.0, -F_e)
    A_w[i, j] = 2.0 * D_w + max(0.0, F_w)
    A_n[i, j] = D_n + max(0.0, -F_n)
    A_s[i, j] = D_s + max(0.0, F_s)
    A_p[i, j] = (
            A_w[i, j] + A_e[i, j] + A_n[i, j] + A_s[i, j] +
            (F_e - F_w) + (F_n - F_s)
    )

    source_x[i, j] = 0.5 * (p[i, j - 1] - p[i, j]) * dx  # 0.5(P_w+P_o)-P_o
    source_y[i, j] = 0.5 * (p[i, j] - p[i - 1, j]) * dy  # P_o - 0.5(P_o+P_n)

    return A_p, A_e, A_w, A_n, A_s, source_x, source_y


# ---------------------------------------------------------------------------
# Iterative solver (Gauss–Seidel with under-relaxation)
# ---------------------------------------------------------------------------

def solve(phi, phi_star, A_p, A_e, A_w, A_n, A_s, source,
          alpha, epsilon, max_inner_iteration, l2_norm_initial, omega):
    # omega = 1.2  # SOR relaxation factor (1 < ω < 2). prev: 1.6

    norm_first = None

    for _ in range(max_inner_iteration):
        l2 = 0.0

        # interior only (avoid ghost cells)
        for i in range(1, n_y + 1):
            for j in range(1, n_x + 1):
                rhs = (
                        A_e[i, j] * phi[i, j + 1] +
                        A_w[i, j] * phi[i, j - 1] +
                        A_n[i, j] * phi[i - 1, j] +
                        A_s[i, j] * phi[i + 1, j] +
                        source[i, j]
                )

                phi_gs = rhs / A_p[i, j]  # Gauss–Seidel result
                phi_new = phi[i, j] + omega * (phi_gs - phi[i, j])  # SOR update

                dphi = phi_new - phi[i, j]
                phi[i, j] = alpha * phi_new + (1 - alpha) * phi_star[i, j]

                l2 += dphi * dphi

        l2 = math.sqrt(l2)

        if norm_first is None:
            norm_first = l2

        if l2 < epsilon:
            break

    if norm_first is None:
        norm_first = l2_norm_initial

    return phi, norm_first


# ---------------------------------------------------------------------------
# Rhie–Chow face velocities (vectorized)
# ---------------------------------------------------------------------------

def face_velocity(u, v, u_face, v_face, p, A_p, alpha_uv):
    # ============================================================
    # U–FACE VELOCITIES (vectorized)
    # ============================================================

    # u-face interior: i = 1..n_y, j = 1..n_x-1
    i_u = slice(1, n_y + 1)

    # j ranges (non-overlapping, correct dimensionality)
    j_u = slice(1, n_x)  # j
    j_u_r1 = slice(2, n_x + 1)  # j+1
    j_u_l1 = slice(0, n_x - 1)  # j-1
    j_u_r2 = slice(3, n_x + 2)  # j+2
    j_u_l2 = slice(1, n_x)  # j (for (p[i,j+2]-p[i,j]))

    # Compute u-face
    u_face[i_u, j_u] = (
            0.5 * (u[i_u, j_u] + u[i_u, j_u_r1]) +
            0.25 * alpha_uv * (p[i_u, j_u_l1] - p[i_u, j_u_r1]) * dy / A_p[i_u, j_u] +
            0.25 * alpha_uv * (p[i_u, j_u_r2] - p[i_u, j_u_l2]) * dy / A_p[i_u, j_u_r1] -
            0.5 * alpha_uv * (1.0 / A_p[i_u, j_u] + 1.0 / A_p[i_u, j_u_r1]) *
            (p[i_u, j_u_r1] - p[i_u, j_u]) * dy
    )

    # ============================================================
    # V–FACE VELOCITIES (vectorized)
    # ============================================================

    # v-face interior corresponds to (i-1, j) for i=2..n_y
    i_v_top = slice(1, n_y)  # i-1
    i_v_bottom = slice(2, n_y + 1)  # i

    j_v = slice(1, n_x + 1)

    # vertical shifted stencils
    i_above = slice(0, n_y - 1)  # i-2
    i_below = slice(3, n_y + 2)  # i+1

    v_face[i_v_top, j_v] = (
            0.5 * (v[i_v_top, j_v] + v[i_v_bottom, j_v]) +
            0.25 * alpha_uv * (p[i_v_top, j_v] - p[i_below, j_v]) * dx / A_p[i_v_bottom, j_v] +
            0.25 * alpha_uv * (p[i_above, j_v] - p[i_v_bottom, j_v]) * dx / A_p[i_v_top, j_v] -
            0.5 * alpha_uv * (1.0 / A_p[i_v_bottom, j_v] + 1.0 / A_p[i_v_top, j_v]) *
            (p[i_v_top, j_v] - p[i_v_bottom, j_v]) * dx
    )

    return u_face, v_face


# ---------------------------------------------------------------------------
# Pressure correction coefficients (vectorized interior)
# ---------------------------------------------------------------------------

def pressure_correction_link_coefficients(u, u_face, v_face,
                                          Ap_p, Ap_e, Ap_w, Ap_n, Ap_s,
                                          source_p, A_p, A_e, A_w, A_n, A_s,
                                          alpha_uv):
    invA = 1.0 / A_p

    # Interior: i=2..n_y-1, j=2..n_x-1
    i_int = slice(2, n_y)
    j_int = slice(2, n_x)

    # Shifted slices
    j_r = slice(3, n_x + 1)  # j+1
    j_l = slice(1, n_x - 1)  # j-1
    i_u = slice(1, n_y - 1)  # i-1
    i_d = slice(3, n_y + 1)  # i+1

    # Coefficients
    Ap_e[i_int, j_int] = 0.5 * alpha_uv * (invA[i_int, j_int] + invA[i_int, j_r]) * dy ** 2
    Ap_w[i_int, j_int] = 0.5 * alpha_uv * (invA[i_int, j_int] + invA[i_int, j_l]) * dy ** 2
    Ap_n[i_int, j_int] = 0.5 * alpha_uv * (invA[i_int, j_int] + invA[i_u, j_int]) * dx ** 2
    Ap_s[i_int, j_int] = 0.5 * alpha_uv * (invA[i_int, j_int] + invA[i_d, j_int]) * dx ** 2

    Ap_p[i_int, j_int] = (
            Ap_e[i_int, j_int] + Ap_w[i_int, j_int]
            + Ap_n[i_int, j_int] + Ap_s[i_int, j_int]
    )

    # Divergence RHS
    source_p[i_int, j_int] = (
            -(u_face[i_int, j_int] - u_face[i_int, j_l]) * dy
            - (v_face[i_u, j_int] - v_face[i_int, j_int]) * dx
    )

    # Top (i = 1)
    i = 1
    for j in range(2, n_x):
        Ap_e[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j + 1]) * (dy ** 2)
        Ap_w[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j - 1]) * (dy ** 2)
        Ap_n[i, j] = 0.0
        Ap_s[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i + 1, j]) * (dx ** 2)
        Ap_p[i, j] = Ap_e[i, j] + Ap_w[i, j] + Ap_n[i, j] + Ap_s[i, j]

        source_p[i, j] = (
                -(u_face[i, j] - u_face[i, j - 1]) * dy -
                (v_face[i - 1, j] - v_face[i, j]) * dx
        )

    # Left (j = 1)
    j = 1
    for i in range(2, n_y):
        Ap_e[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j + 1]) * (dy ** 2)
        Ap_w[i, j] = 0.0
        Ap_n[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i - 1, j]) * (dx ** 2)
        Ap_s[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i + 1, j]) * (dx ** 2)
        Ap_p[i, j] = Ap_e[i, j] + Ap_w[i, j] + Ap_n[i, j] + Ap_s[i, j]

        source_p[i, j] = (
                -(u_face[i, j] - u_face[i, j - 1]) * dy -
                (v_face[i - 1, j] - v_face[i, j]) * dx
        )

    # Right (j = n_x)
    j = n_x
    for i in range(2, n_y):
        Ap_e[i, j] = 0.0
        Ap_w[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j - 1]) * (dy ** 2)
        Ap_n[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i - 1, j]) * (dx ** 2)
        Ap_s[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i + 1, j]) * (dx ** 2)
        Ap_p[i, j] = Ap_e[i, j] + Ap_w[i, j] + Ap_n[i, j] + Ap_s[i, j]

        source_p[i, j] = (
                -(u_face[i, j] - u_face[i, j - 1]) * dy -
                (v_face[i - 1, j] - v_face[i, j]) * dx
        )

    # Bottom (i = n_y)
    i = n_y
    for j in range(2, n_x):
        Ap_e[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j + 1]) * (dy ** 2)
        Ap_w[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j - 1]) * (dy ** 2)
        Ap_n[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i - 1, j]) * (dx ** 2)
        Ap_s[i, j] = 0.0
        Ap_p[i, j] = Ap_e[i, j] + Ap_w[i, j] + Ap_n[i, j] + Ap_s[i, j]

        source_p[i, j] = (
                -(u_face[i, j] - u_face[i, j - 1]) * dy -
                (v_face[i - 1, j] - v_face[i, j]) * dx
        )

    # Corners (same as your original)
    # top left
    i = 1
    j = 1
    Ap_e[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j + 1]) * (dy ** 2)
    Ap_w[i, j] = 0.0
    Ap_n[i, j] = 0.0
    Ap_s[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i + 1, j]) * (dx ** 2)
    Ap_p[i, j] = Ap_e[i, j] + Ap_w[i, j] + Ap_n[i, j] + Ap_s[i, j]
    source_p[i, j] = (
            -(u_face[i, j] - u_face[i, j - 1]) * dy -
            (v_face[i - 1, j] - v_face[i, j]) * dx
    )

    # top right
    i = 1
    j = n_x
    Ap_e[i, j] = 0.0
    Ap_w[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j - 1]) * (dy ** 2)
    Ap_n[i, j] = 0.0
    Ap_s[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i + 1, j]) * (dx ** 2)
    Ap_p[i, j] = Ap_e[i, j] + Ap_w[i, j] + Ap_n[i, j] + Ap_s[i, j]
    source_p[i, j] = (
            -(u_face[i, j] - u_face[i, j - 1]) * dy -
            (v_face[i - 1, j] - v_face[i, j]) * dx
    )

    # bottom left
    i = n_y
    j = 1
    Ap_e[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j + 1]) * (dy ** 2)
    Ap_w[i, j] = 0.0
    Ap_n[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i - 1, j]) * (dx ** 2)
    Ap_s[i, j] = 0.0
    Ap_p[i, j] = Ap_e[i, j] + Ap_w[i, j] + Ap_n[i, j] + Ap_s[i, j]
    source_p[i, j] = (
            -(u_face[i, j] - u_face[i, j - 1]) * dy -
            (v_face[i - 1, j] - v_face[i, j]) * dx
    )

    # bottom right
    i = n_y
    j = n_x
    Ap_e[i, j] = 0.0
    Ap_w[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j - 1]) * (dy ** 2)
    Ap_n[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i - 1, j]) * (dx ** 2)
    Ap_s[i, j] = 0.0
    Ap_p[i, j] = Ap_e[i, j] + Ap_w[i, j] + Ap_n[i, j] + Ap_s[i, j]
    source_p[i, j] = (
            -(u_face[i, j] - u_face[i, j - 1]) * dy -
            (v_face[i - 1, j] - v_face[i, j]) * dx
    )

    return Ap_p, Ap_e, Ap_w, Ap_n, Ap_s, source_p


# ---------------------------------------------------------------------------
# Pressure correction
# ---------------------------------------------------------------------------

def correct_pressure(p_star, p, p_prime, alpha_p):
    p_star = p + alpha_p * p_prime

    # BCs for ghost cells

    # top wall
    p_star[0, 1:n_x + 1] = p_star[1, 1:n_x + 1]
    # left wall
    p_star[1:n_y + 1, 0] = p_star[1:n_y + 1, 1]
    # right wall
    p_star[1:n_y + 1, n_x + 1] = p_star[1:n_y + 1, n_x]
    # bottom wall
    p_star[n_y + 1, 1:n_x + 1] = p_star[n_y, 1:n_x + 1]

    # corners
    p_star[0, 0] = (p_star[1, 2] + p_star[0, 1] + p_star[1, 0]) / 3.0
    p_star[0, n_x + 1] = (p_star[0, n_x] + p_star[1, n_x] + p_star[1, n_x + 1]) / 3.0
    p_star[n_y + 1, 0] = (p_star[n_y, 0] + p_star[n_y, 1] + p_star[n_y + 1, 1]) / 3.0
    p_star[n_y + 1, n_x + 1] = (
                                       p_star[n_y, n_x + 1] + p_star[n_y + 1, n_x] + p_star[n_y, n_x]
                               ) / 3.0

    return p_star


# ---------------------------------------------------------------------------
# Correct cell-centered velocities
# ---------------------------------------------------------------------------

def correct_cell_center_velocity(u, v, u_star, v_star, p_prime, A_p, alpha_uv):
    # u velocity (interior)
    for i in range(1, n_y + 1):
        for j in range(2, n_x):
            u_star[i, j] = u[i, j] + 0.5 * alpha_uv * (
                    p_prime[i, j - 1] - p_prime[i, j + 1]
            ) * dy / A_p[i, j]

    # left boundary
    j = 1
    for i in range(1, n_y + 1):
        u_star[i, j] = u[i, j] + 0.5 * alpha_uv * (
                p_prime[i, j] - p_prime[i, j + 1]
        ) * dy / A_p[i, j]

    # right boundary
    j = n_x
    for i in range(1, n_y + 1):
        u_star[i, j] = u[i, j] + 0.5 * alpha_uv * (
                p_prime[i, j - 1] - p_prime[i, j]
        ) * dy / A_p[i, j]

    # v velocity (interior)
    for i in range(2, n_y):
        for j in range(1, n_x + 1):
            v_star[i, j] = v[i, j] + 0.5 * alpha_uv * (
                    p_prime[i + 1, j] - p_prime[i - 1, j]
            ) * dx / A_p[i, j]

    # top
    i = 1
    for j in range(1, n_x + 1):
        v_star[i, j] = v[i, j] + 0.5 * alpha_uv * (
                p_prime[i + 1, j] - p_prime[i, j]
        ) * dx / A_p[i, j]

    # bottom
    i = n_y
    for j in range(1, n_x + 1):
        v_star[i, j] = v[i, j] + 0.5 * alpha_uv * (
                p_prime[i, j] - p_prime[i - 1, j]
        ) * dx / A_p[i, j]

    return u_star, v_star


# ---------------------------------------------------------------------------
# Correct face velocities with pressure correction
# ---------------------------------------------------------------------------

def correct_face_velocity(u_face, v_face, p_prime, A_p, alpha_uv):
    # u faces
    for i in range(1, n_y + 1):
        for j in range(1, n_x):
            u_face[i, j] = u_face[i, j] + 0.5 * alpha_uv * (
                    1.0 / A_p[i, j] + 1.0 / A_p[i, j + 1]
            ) * (p_prime[i, j] - p_prime[i, j + 1]) * dy

    # v faces
    for i in range(2, n_y + 1):
        for j in range(1, n_x + 1):
            v_face[i - 1, j] = v_face[i - 1, j] + 0.5 * alpha_uv * (
                    1.0 / A_p[i, j] + 1.0 / A_p[i - 1, j]
            ) * (p_prime[i, j] - p_prime[i - 1, j]) * dx

    return u_face, v_face


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def post_processing(u_star, v_star, p_star, X, Y, x, y):
    # Extract only interior cells (remove ghost cells)
    u_interior = u_star[1:n_y + 1, 1:n_x + 1]
    v_interior = v_star[1:n_y + 1, 1:n_x + 1]
    p_interior = p_star[1:n_y + 1, 1:n_x + 1]
    x_interior = x[1:n_x + 1]  # Remove ghost cells
    y_interior = y[1:n_y + 1]  # Remove ghost cells

    # Create meshgrid for interior cells only
    X_interior, Y_interior = np.meshgrid(x_interior, y_interior)

    # u velocity contours
    plt.figure(1)
    plt.contourf(X_interior, Y_interior, np.flipud(u_interior), levels=50, cmap='jet')
    plt.colorbar()
    plt.title('U contours')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # v velocity contours
    plt.figure(2)
    plt.contourf(X_interior, Y_interior, np.flipud(v_interior), levels=50, cmap='jet')
    plt.colorbar()
    plt.title('V contours')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # pressure contours
    plt.figure(3)
    plt.contourf(X_interior, Y_interior, np.flipud(p_interior), levels=50, cmap='jet')
    plt.colorbar()
    plt.title('P contours')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # u centerline velocity
    plt.figure(4)
    plt.plot(1 - y_interior, u_interior[:, round(n_x / 2)])
    plt.xlabel('y')
    plt.ylabel('u')
    plt.title('U centerline velocity')
    plt.grid(True)
    plt.show()

    # v centerline velocity
    plt.figure(5)
    plt.plot(x_interior, v_interior[round(n_y / 2), :])
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title('V centerline velocity')
    plt.grid(True)
    plt.show()

    # streamlines
    plt.figure(6)
    u_plot = np.flipud(u_interior)
    v_plot = np.flipud(v_interior)
    plt.streamplot(X_interior, Y_interior, u_plot, v_plot, density=1.8, linewidth=1, arrowsize=1)
    plt.title('Streamlines')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{n_x}x{n_y}_A4_v5_v4_results.png', dpi=150)
    plt.show()


def fix_pressure_reference(p_prime):
    """
    Fix pressure at reference point to remove null space.
    For lid-driven cavity, typically fix at a corner.
    """
    # Fix pressure at bottom-left interior point
    p_ref = p_prime[n_y, 1]  # Store value at reference
    p_prime[1:n_y + 1, 1:n_x + 1] -= p_ref  # Subtract from all interior
    return p_prime


def get_relaxation_factors(n, l2_norm_p, l2_norm_p_prev):
    """Adaptive relaxation based on iteration count and convergence behavior"""

    if n <= 30:
        # Very conservative startup
        alpha_uv = 0.5
        alpha_p = 0.2
        omega_uv = 1.0
        omega_p = 1.0
    elif n <= 100:
        # Moderate relaxation
        alpha_uv = 0.7
        alpha_p = 0.3
        omega_uv = 1.0
        omega_p = 1.2
    else:
        # More aggressive once stable
        # But back off if pressure residual increases
        if l2_norm_p_prev > 0 and l2_norm_p > 1.2 * l2_norm_p_prev:
            # Residual increased - reduce relaxation
            alpha_uv = 0.6
            alpha_p = 0.2
            omega_uv = 1.0
            omega_p = 1.0
        else:
            alpha_uv = 0.7
            alpha_p = 0.4  # Can be more aggressive for pressure
            omega_uv = 1.1
            omega_p = 1.3

    return alpha_uv, alpha_p, omega_uv, omega_p


def momentum_predictor_step(u, v, u_star, v_star):
    """
    Extrapolate velocity from previous iterations
    u^n+1 ≈ 2*u^n - u^n-1
    """
    u_pred = 1.5 * u - 0.5 * u_star
    v_pred = 1.5 * v - 0.5 * v_star
    return u_pred, v_pred


# ---------------------------------------------------------------------------
# MAIN SETUP
# ---------------------------------------------------------------------------

# Primitive variables (with ghost cells)
u = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
u_star = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)

v = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
v_star = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)

p_star = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
p = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
p_prime = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)

# Momentum link coefficients
A_p = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
A_e = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
A_w = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
A_s = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
A_n = np.ones((n_y + 2, n_x + 2), dtype=np.float64)

# Pressure correction coefficients
Ap_p = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
Ap_e = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
Ap_w = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
Ap_s = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
Ap_n = np.ones((n_y + 2, n_x + 2), dtype=np.float64)

# Source terms
source_x = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
source_y = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
source_p = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)

# Face velocities
u_face = np.zeros((n_y + 2, n_x + 1), dtype=np.float64)
v_face = np.zeros((n_y + 1, n_x + 2), dtype=np.float64)

# Grid (cell centers including ghost nodes)
x = np.array([0.0], dtype=np.float64)
y = np.array([0.0], dtype=np.float64)

x = np.append(x, np.linspace(dx / 2, 1 - dx / 2, n_x))
x = np.append(x, [1.0])

y = np.append(y, np.linspace(dy / 2, 1 - dy / 2, n_y))
y = np.append(y, [1.0])

X, Y = np.meshgrid(x, y)

# Lid boundary condition: top lid u = 1
u[0, 1:n_x + 1] = 1.0
u_star[0, 1:n_x + 1] = 1.0
u_face[0, 1:n_x] = 1.0

# Solver parameters
l2_norm_x = 0.0
l2_norm_y = 0.0
l2_norm_p = 0.0

alpha_uv = 0.3  # prev 0.25, 0.7
epsilon_uv = 1e-4
max_inner_iteration_uv = 20
omega_uv = 1.0

max_inner_iteration_p = 300
dummy_alpha_p = 1.0
epsilon_p = 1e-5
alpha_p = 0.1  # prev 0.1, 0.3
omega_p = 1.0

max_outer_iteration = 2000

# ---------------------------------------------------------------------------
# SIMPLE outer loop
# ---------------------------------------------------------------------------

for n in range(1, max_outer_iteration + 1):

    l2_norm_p_prev = l2_norm_p if n > 1 else 1.0
    # alpha_uv, alpha_p, omega_uv, omega_p = get_relaxation_factors(n, l2_norm_p, l2_norm_p_prev)

    if n > 2:
        u_pred, v_pred = momentum_predictor_step(u, v, u_star, v_star)
        u = 0.5 * u + 0.5 * u_pred  # Blend with predictor
        v = 0.5 * v + 0.5 * v_pred

    A_p, A_e, A_w, A_n, A_s, source_x, source_y = momentum_link_coefficients(
        u_star, u_face, v_face, p, source_x, source_y, A_p, A_e, A_w, A_n, A_s
    )

    # Solve u-momentum
    u, l2_norm_x = solve(
        u, u_star, A_p, A_e, A_w, A_n, A_s,
        source_x, alpha_uv, epsilon_uv, max_inner_iteration_uv, l2_norm_x, omega_uv
    )

    # Solve v-momentum
    v, l2_norm_y = solve(
        v, v_star, A_p, A_e, A_w, A_n, A_s,
        source_y, alpha_uv, epsilon_uv, max_inner_iteration_uv, l2_norm_y, omega_uv
    )

    # Rhie–Chow face velocities
    u_face, v_face = face_velocity(u, v, u_face, v_face, p, A_p, alpha_uv)

    # Pressure correction coefficients
    Ap_p, Ap_e, Ap_w, Ap_n, Ap_s, source_p = pressure_correction_link_coefficients(
        u, u_face, v_face, Ap_p, Ap_e, Ap_w, Ap_n, Ap_s,
        source_p, A_p, A_e, A_w, A_n, A_s, alpha_uv
    )

    # Solve for pressure correction p'
    p_prime, l2_norm_p = solve(
        p_prime, p_prime, Ap_p, Ap_e, Ap_w, Ap_n, Ap_s,
        source_p, dummy_alpha_p, epsilon_p, max_inner_iteration_p, l2_norm_p, omega_p
    )
    p_prime = fix_pressure_reference(p_prime)
    print("sum(source_p = ", np.sum(source_p))
    # Correct pressure
    p_star = correct_pressure(p_star, p, p_prime, alpha_p)

    # Correct cell-centered velocities
    u_star, v_star = correct_cell_center_velocity(u, v, u_star, v_star, p_prime, A_p, alpha_uv)

    # Correct face velocities with pressure correction
    u_face, v_face = correct_face_velocity(u_face, v_face, p_prime, A_p, alpha_uv)

    # Update pressure for next SIMPLE iteration
    p = np.copy(p_star)

    if n % 50 == 0:
        p_mean = p[1:n_y + 1, 1:n_x + 1].mean()
        p[1:n_y + 1, 1:n_x + 1] -= p_mean

    print(f"Iter {n:4d}: l2_u = {l2_norm_x: .3e}, l2_v = {l2_norm_y: .3e}, l2_p = {l2_norm_p: .3e}")

    max_div = np.abs(source_p[1:n_y + 1, 1:n_x + 1]).max()
    max_u = np.abs(u[1:n_y + 1, 1:n_x + 1]).max()
    max_v = np.abs(v[1:n_y + 1, 1:n_x + 1]).max()
    max_p = np.abs(p[1:n_y + 1, 1:n_x + 1]).max()

    print(f"  Max div: {max_div:.3e}, Max u: {max_u:.3e}, "
          f"Max v: {max_v:.3e}, Max p: {max_p:.3e}")

    # Check for divergence
    if max_u > 10 or max_v > 10 or max_p > 1000:
        print("SOLUTION DIVERGING - stopping")
        break

    if (l2_norm_x < 1e-4) and (l2_norm_y < 1e-4) and (l2_norm_p < 1e-4) and (max_div < 1e-5):
        print("Converged!")
        break

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

post_processing(u_star, v_star, p_star, X, Y, x, y)
