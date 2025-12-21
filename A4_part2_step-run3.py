# worse inner iteration tolerance

import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from scipy.interpolate import RectBivariateSpline
import os
import time

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

n_x = 320
n_y = 320

dx = 1.0 / n_x
dy = 1.0 / n_y

Re = 200

# Step geometry parameters (based on Figure 2)
# Step occupies bottom-left corner: 0 <= x <= 0.25, 0 <= y <= 0.25
step_x_fraction = 0.25  # Step extends 25% in x-direction
step_y_fraction = 0.25  # Step extends 25% in y-direction

step_x_cells = int(n_x * step_x_fraction)
step_y_cells = int(n_y * step_y_fraction)

print(f"Step geometry: {step_x_cells} x {step_y_cells} cells")

# Create a mask for blocked (solid) cells
# blocked[i,j] = True means cell (i,j) is inside the solid step
blocked = np.zeros((n_y + 2, n_x + 2), dtype=bool)

# Block bottom-left corner cells (interior cells only)
# In our indexing: i=1 is top, i=n_y is bottom
# j=1 is left, j=n_x is right
# Bottom-left corner: i from (n_y - step_y_cells + 1) to n_y, j from 1 to step_x_cells
blocked[n_y - step_y_cells + 1:n_y + 1, 1:step_x_cells + 1] = True

print(f"Blocked cells: {np.sum(blocked)} total")


# ---------------------------------------------------------------------------
# Momentum link coefficients with step geometry
# ---------------------------------------------------------------------------

def momentum_link_coefficients(u_star, u_face, v_face, p, source_x, source_y,
                               A_p, A_e, A_w, A_n, A_s, blocked):
    """
    Build momentum equation coefficients and source terms.
    Modified to handle blocked cells (solid step).
    """

    D_e = dy / (dx * Re)
    D_w = dy / (dx * Re)
    D_n = dx / (dy * Re)
    D_s = dx / (dy * Re)

    # Initialize all to zero
    A_p[:, :] = 1.0
    A_e[:, :] = 0.0
    A_w[:, :] = 0.0
    A_n[:, :] = 0.0
    A_s[:, :] = 0.0
    source_x[:, :] = 0.0
    source_y[:, :] = 0.0

    # Loop over all interior cells
    for i in range(1, n_y + 1):
        for j in range(1, n_x + 1):

            # Skip blocked cells
            if blocked[i, j]:
                A_p[i, j] = 1.0
                A_e[i, j] = 0.0
                A_w[i, j] = 0.0
                A_n[i, j] = 0.0
                A_s[i, j] = 0.0
                source_x[i, j] = 0.0
                source_y[i, j] = 0.0
                continue

            # Check if neighbors are blocked
            blocked_e = blocked[i, j + 1] if j < n_x else False
            blocked_w = blocked[i, j - 1] if j > 1 else False
            blocked_n = blocked[i - 1, j] if i > 1 else False
            blocked_s = blocked[i + 1, j] if i < n_y else False

            # Fluxes
            F_e = dy * u_face[i, j] if not blocked_e else 0.0
            F_w = dy * u_face[i, j - 1] if not blocked_w else 0.0
            F_n = dx * v_face[i - 1, j] if not blocked_n else 0.0
            F_s = dx * v_face[i, j] if not blocked_s else 0.0

            # Coefficients (use 2*D for walls and blocked neighbors)
            if j == n_x or blocked_e:
                A_e[i, j] = 0.0
            elif j == 1:
                A_e[i, j] = D_e + max(0.0, -F_e)
            else:
                A_e[i, j] = D_e + max(0.0, -F_e)

            if j == 1 or blocked_w:
                A_w[i, j] = 2.0 * D_w + max(0.0, F_w)
            elif j == n_x:
                A_w[i, j] = 2.0 * D_w + max(0.0, F_w)
            else:
                A_w[i, j] = D_w + max(0.0, F_w)

            if i == 1 or blocked_n:
                A_n[i, j] = 2.0 * D_n + max(0.0, -F_n)
            elif i == n_y:
                A_n[i, j] = D_n + max(0.0, -F_n)
            else:
                A_n[i, j] = D_n + max(0.0, -F_n)

            if i == n_y or blocked_s:
                A_s[i, j] = 2.0 * D_s + max(0.0, F_s)
            elif i == 1:
                A_s[i, j] = D_s + max(0.0, F_s)
            else:
                A_s[i, j] = D_s + max(0.0, F_s)

            A_p[i, j] = (
                    A_w[i, j] + A_e[i, j] + A_n[i, j] + A_s[i, j] +
                    (F_e - F_w) + (F_n - F_s)
            )

            # Pressure source terms
            if j == 1:
                source_x[i, j] = 0.5 * (p[i, j] - p[i, j + 1]) * dx
            elif j == n_x:
                source_x[i, j] = 0.5 * (p[i, j - 1] - p[i, j]) * dx
            else:
                source_x[i, j] = 0.5 * (p[i, j - 1] - p[i, j + 1]) * dx

            if i == 1:
                source_y[i, j] = 0.5 * (p[i + 1, j] - p[i, j]) * dy
            elif i == n_y:
                source_y[i, j] = 0.5 * (p[i, j] - p[i - 1, j]) * dy
            else:
                source_y[i, j] = 0.5 * (p[i + 1, j] - p[i - 1, j]) * dy

    return A_p, A_e, A_w, A_n, A_s, source_x, source_y


# ---------------------------------------------------------------------------
# Iterative solver (Gauss–Seidel with under-relaxation)
# ---------------------------------------------------------------------------

def solve(phi, phi_star, A_p, A_e, A_w, A_n, A_s, source,
          alpha, epsilon, max_inner_iteration, l2_norm_initial, omega, blocked):
    norm_first = None

    for _ in range(max_inner_iteration):
        l2 = 0.0

        for i in range(1, n_y + 1):
            for j in range(1, n_x + 1):

                # Skip blocked cells
                if blocked[i, j]:
                    phi[i, j] = 0.0
                    continue

                rhs = (
                        A_e[i, j] * phi[i, j + 1] +
                        A_w[i, j] * phi[i, j - 1] +
                        A_n[i, j] * phi[i - 1, j] +
                        A_s[i, j] * phi[i + 1, j] +
                        source[i, j]
                )

                phi_gs = rhs / A_p[i, j]
                phi_new = phi[i, j] + omega * (phi_gs - phi[i, j])

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
# Rhie–Chow face velocities
# ---------------------------------------------------------------------------

def face_velocity(u, v, u_face, v_face, p, A_p, alpha_uv, blocked):
    """Modified to handle blocked cells"""

    # U-face velocities
    for i in range(1, n_y + 1):
        for j in range(1, n_x):
            # Check if either adjacent cell is blocked
            if blocked[i, j] or blocked[i, j + 1]:
                u_face[i, j] = 0.0
                continue

            u_face[i, j] = (
                    0.5 * (u[i, j] + u[i, j + 1]) +
                    0.25 * alpha_uv * (p[i, j - 1] - p[i, j + 1]) * dy / A_p[i, j] +
                    0.25 * alpha_uv * (p[i, j + 2] - p[i, j]) * dy / A_p[i, j + 1] -
                    0.5 * alpha_uv * (1.0 / A_p[i, j] + 1.0 / A_p[i, j + 1]) *
                    (p[i, j + 1] - p[i, j]) * dy
            )

    # V-face velocities
    for i in range(1, n_y):
        for j in range(1, n_x + 1):
            # Check if either adjacent cell is blocked
            if blocked[i, j] or blocked[i + 1, j]:
                v_face[i, j] = 0.0
                continue

            v_face[i, j] = (
                    0.5 * (v[i, j] + v[i + 1, j]) +
                    0.25 * alpha_uv * (p[i, j] - p[i + 2, j]) * dx / A_p[i + 1, j] +
                    0.25 * alpha_uv * (p[i - 1, j] - p[i + 1, j]) * dx / A_p[i, j] -
                    0.5 * alpha_uv * (1.0 / A_p[i + 1, j] + 1.0 / A_p[i, j]) *
                    (p[i, j] - p[i + 1, j]) * dx
            )

    return u_face, v_face


# ---------------------------------------------------------------------------
# Pressure correction coefficients
# ---------------------------------------------------------------------------

def pressure_correction_link_coefficients(u, u_face, v_face,
                                          Ap_p, Ap_e, Ap_w, Ap_n, Ap_s,
                                          source_p, A_p, A_e, A_w, A_n, A_s,
                                          alpha_uv, blocked):
    invA = 1.0 / A_p

    for i in range(1, n_y + 1):
        for j in range(1, n_x + 1):

            if blocked[i, j]:
                Ap_p[i, j] = 1.0
                Ap_e[i, j] = 0.0
                Ap_w[i, j] = 0.0
                Ap_n[i, j] = 0.0
                Ap_s[i, j] = 0.0
                source_p[i, j] = 0.0
                continue

            # Check blocked neighbors
            blocked_e = blocked[i, j + 1] if j < n_x else False
            blocked_w = blocked[i, j - 1] if j > 1 else False
            blocked_n = blocked[i - 1, j] if i > 1 else False
            blocked_s = blocked[i + 1, j] if i < n_y else False

            # Coefficients
            if j < n_x and not blocked_e:
                Ap_e[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j + 1]) * dy ** 2
            else:
                Ap_e[i, j] = 0.0

            if j > 1 and not blocked_w:
                Ap_w[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i, j - 1]) * dy ** 2
            else:
                Ap_w[i, j] = 0.0

            if i > 1 and not blocked_n:
                Ap_n[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i - 1, j]) * dx ** 2
            else:
                Ap_n[i, j] = 0.0

            if i < n_y and not blocked_s:
                Ap_s[i, j] = 0.5 * alpha_uv * (invA[i, j] + invA[i + 1, j]) * dx ** 2
            else:
                Ap_s[i, j] = 0.0

            Ap_p[i, j] = Ap_e[i, j] + Ap_w[i, j] + Ap_n[i, j] + Ap_s[i, j]

            # Divergence source
            source_p[i, j] = (
                    -(u_face[i, j] - u_face[i, j - 1]) * dy -
                    (v_face[i - 1, j] - v_face[i, j]) * dx
            )

    return Ap_p, Ap_e, Ap_w, Ap_n, Ap_s, source_p


# ---------------------------------------------------------------------------
# Pressure correction
# ---------------------------------------------------------------------------

def correct_pressure(p_star, p, p_prime, alpha_p, blocked):
    p_star = p + alpha_p * p_prime

    # BCs for ghost cells (same as before)
    p_star[0, 1:n_x + 1] = p_star[1, 1:n_x + 1]
    p_star[1:n_y + 1, 0] = p_star[1:n_y + 1, 1]
    p_star[1:n_y + 1, n_x + 1] = p_star[1:n_y + 1, n_x]
    p_star[n_y + 1, 1:n_x + 1] = p_star[n_y, 1:n_x + 1]

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

def correct_cell_center_velocity(u, v, u_star, v_star, p_prime, A_p, alpha_uv, blocked):
    for i in range(1, n_y + 1):
        for j in range(1, n_x + 1):

            if blocked[i, j]:
                u_star[i, j] = 0.0
                v_star[i, j] = 0.0
                continue

            # u correction
            if j == 1:
                u_star[i, j] = u[i, j] + 0.5 * alpha_uv * (
                        p_prime[i, j] - p_prime[i, j + 1]
                ) * dy / A_p[i, j]
            elif j == n_x:
                u_star[i, j] = u[i, j] + 0.5 * alpha_uv * (
                        p_prime[i, j - 1] - p_prime[i, j]
                ) * dy / A_p[i, j]
            else:
                u_star[i, j] = u[i, j] + 0.5 * alpha_uv * (
                        p_prime[i, j - 1] - p_prime[i, j + 1]
                ) * dy / A_p[i, j]

            # v correction
            if i == 1:
                v_star[i, j] = v[i, j] + 0.5 * alpha_uv * (
                        p_prime[i + 1, j] - p_prime[i, j]
                ) * dx / A_p[i, j]
            elif i == n_y:
                v_star[i, j] = v[i, j] + 0.5 * alpha_uv * (
                        p_prime[i, j] - p_prime[i - 1, j]
                ) * dx / A_p[i, j]
            else:
                v_star[i, j] = v[i, j] + 0.5 * alpha_uv * (
                        p_prime[i + 1, j] - p_prime[i - 1, j]
                ) * dx / A_p[i, j]

    return u_star, v_star


# ---------------------------------------------------------------------------
# Correct face velocities
# ---------------------------------------------------------------------------

def correct_face_velocity(u_face, v_face, p_prime, A_p, alpha_uv, blocked):
    for i in range(1, n_y + 1):
        for j in range(1, n_x):
            if blocked[i, j] or blocked[i, j + 1]:
                u_face[i, j] = 0.0
                continue

            u_face[i, j] = u_face[i, j] + 0.5 * alpha_uv * (
                    1.0 / A_p[i, j] + 1.0 / A_p[i, j + 1]
            ) * (p_prime[i, j] - p_prime[i, j + 1]) * dy

    for i in range(1, n_y):
        for j in range(1, n_x + 1):
            if blocked[i, j] or blocked[i + 1, j]:
                v_face[i, j] = 0.0
                continue

            v_face[i, j] = v_face[i, j] + 0.5 * alpha_uv * (
                    1.0 / A_p[i + 1, j] + 1.0 / A_p[i, j]
            ) * (p_prime[i + 1, j] - p_prime[i, j]) * dx

    return u_face, v_face


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def post_processing(u_star, v_star, p_star, X, Y, x, y, blocked,
                    Re=None,
                    step_x_fraction=None,
                    step_y_fraction=None,
                    save=False,
                    prefix="cavity"):
    """
    Expanded post-processing to match reference implementation:
    - Masks blocked cells
    - Proper interior extraction
    - U, V, P contours
    - Centerline velocity plots
    - Streamlines with solid step
    """

    # ------------------------------------------------------------------
    # Extract interior (remove ghost cells)
    # ------------------------------------------------------------------
    u_int = u_star[1:n_y + 1, 1:n_x + 1].copy()
    v_int = v_star[1:n_y + 1, 1:n_x + 1].copy()
    p_int = p_star[1:n_y + 1, 1:n_x + 1].copy()
    blocked_int = blocked[1:n_y + 1, 1:n_x + 1]

    x_int = x[1:n_x + 1]
    y_int = y[1:n_y + 1]
    X_int, Y_int = np.meshgrid(x_int, y_int)

    # ------------------------------------------------------------------
    # Mask blocked cells
    # ------------------------------------------------------------------
    u_masked = np.ma.masked_where(blocked_int, u_int)
    v_masked = np.ma.masked_where(blocked_int, v_int)
    p_masked = np.ma.masked_where(blocked_int, p_int)

    # ==================================================================
    # U velocity contours
    # ==================================================================
    plt.figure(figsize=(7, 6))
    plt.contourf(X_int, Y_int, np.flipud(u_masked), levels=50, cmap='jet')
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('U velocity contours')
    plt.axis('equal')
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_u_contours.png", dpi=200)
    plt.show()

    # ==================================================================
    # V velocity contours
    # ==================================================================
    plt.figure(figsize=(7, 6))
    plt.contourf(X_int, Y_int, np.flipud(v_masked), levels=50, cmap='jet')
    plt.colorbar(label='v')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('V velocity contours')
    plt.axis('equal')
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_v_contours.png", dpi=200)
    plt.show()

    # ==================================================================
    # Pressure contours + step outline
    # ==================================================================
    plt.figure(figsize=(7, 6))
    plt.contourf(X_int, Y_int, np.flipud(p_masked), levels=50, cmap='jet')
    plt.colorbar(label='p')

    # Draw step geometry if provided
    if step_x_fraction is not None and step_y_fraction is not None:
        sx = step_x_fraction
        sy = step_y_fraction
        plt.plot([0, sx, sx, 0, 0],
                 [0, 0, sy, sy, 0],
                 'k-', linewidth=2)
        plt.fill([0, sx, sx, 0],
                 [0, 0, sy, sy],
                 color='gray', alpha=0.3)

    title = "Pressure contours"
    if Re is not None:
        title += f" (Re={Re})"
    plt.title(title)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_pressure_contours.png", dpi=200)
    plt.show()

    # ==================================================================
    # U centerline velocity (vertical mid-plane)
    # ==================================================================
    j_mid = n_x // 2
    plt.figure()
    plt.plot(1.0 - y_int, u_int[:, j_mid])
    plt.xlabel('y')
    plt.ylabel('u')
    plt.title('U centerline velocity')
    plt.grid(True)
    if save:
        plt.savefig(f"{prefix}_u_centerline.png", dpi=200)
    plt.show()

    # ==================================================================
    # V centerline velocity (horizontal mid-plane)
    # ==================================================================
    i_mid = n_y // 2
    plt.figure()
    plt.plot(x_int, v_int[i_mid, :])
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title('V centerline velocity')
    plt.grid(True)
    if save:
        plt.savefig(f"{prefix}_v_centerline.png", dpi=200)
    plt.show()

    # ==================================================================
    # Streamlines (blocked cells zeroed)
    # ==================================================================
    plt.figure(figsize=(7, 6))

    u_plot = np.flipud(u_masked)
    v_plot = np.flipud(v_masked)
    mask_plot = np.flipud(blocked_int)

    u_filled = np.where(mask_plot, 0.0, u_plot)
    v_filled = np.where(mask_plot, 0.0, v_plot)

    plt.streamplot(X_int, Y_int,
                   u_filled, v_filled,
                   density=2.0,
                   linewidth=1.0,
                   arrowsize=1.0)

    # Step outline again
    if step_x_fraction is not None and step_y_fraction is not None:
        plt.plot([0, sx, sx, 0, 0],
                 [0, 0, sy, sy, 0],
                 'k-', linewidth=2)

    title = "Streamlines"
    if Re is not None:
        title += f" (Re={Re})"
    plt.title(title)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_streamlines.png", dpi=200)
    plt.show()



def fix_pressure_reference(p_prime, blocked):
    """Fix pressure at reference point"""
    # Find first non-blocked interior cell for reference
    for i in range(1, n_y + 1):
        for j in range(1, n_x + 1):
            if not blocked[i, j]:
                p_ref = p_prime[i, j]
                p_prime[1:n_y + 1, 1:n_x + 1] -= p_ref
                return p_prime
    return p_prime


# ---------------------------------------------------------------------------
# MAIN SETUP
# ---------------------------------------------------------------------------

print(f"Solving lid-driven cavity with step at Re={Re}")
print(f"Grid: {n_x} x {n_y}")
print(f"Step size: {step_x_fraction} x {step_y_fraction}")

# Initialize arrays
u = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
v = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
p = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)

u_star = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
v_star = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
p_star = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
p_prime = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)

# Lid boundary condition
u[0, 1:n_x + 1] = 1.0
u_star[0, 1:n_x + 1] = 1.0

# Coefficients
A_p = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
A_e = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
A_w = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
A_s = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
A_n = np.ones((n_y + 2, n_x + 2), dtype=np.float64)

Ap_p = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
Ap_e = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
Ap_w = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
Ap_s = np.ones((n_y + 2, n_x + 2), dtype=np.float64)
Ap_n = np.ones((n_y + 2, n_x + 2), dtype=np.float64)

source_x = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
source_y = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
source_p = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)

u_face = np.zeros((n_y + 2, n_x + 1), dtype=np.float64)
v_face = np.zeros((n_y + 1, n_x + 2), dtype=np.float64)

# Grid
x = np.array([0.0], dtype=np.float64)
y = np.array([0.0], dtype=np.float64)

x = np.append(x, np.linspace(dx / 2, 1 - dx / 2, n_x))
x = np.append(x, [1.0])

y = np.append(y, np.linspace(dy / 2, 1 - dy / 2, n_y))
y = np.append(y, [1.0])

X, Y = np.meshgrid(x, y)

u[0, 1:n_x + 1] = 1.0
u_star[0, 1:n_x + 1] = 1.0
u_face[0, 1:n_x] = 1.0

# Solver parameters
l2_norm_x = 0.0
l2_norm_y = 0.0
l2_norm_p = 0.0

alpha_uv = 0.05
epsilon_uv = 1e-5
max_inner_iteration_uv = 25
omega_uv = 0.8

max_inner_iteration_p = 125
dummy_alpha_p = 1.0
epsilon_p = 1e-6
alpha_p = 0.004
omega_p = 0.8

max_outer_iteration = 2000


time_start = time.time()

# ---------------------------------------------------------------------------
# SIMPLE outer loop
# ---------------------------------------------------------------------------

for n in range(1, max_outer_iteration + 1):

    iter_start = time.time()

    A_p, A_e, A_w, A_n, A_s, source_x, source_y = momentum_link_coefficients(
        u_star, u_face, v_face, p, source_x, source_y, A_p, A_e, A_w, A_n, A_s, blocked
    )

    u, l2_norm_x = solve(
        u, u_star, A_p, A_e, A_w, A_n, A_s,
        source_x, alpha_uv, epsilon_uv, max_inner_iteration_uv, l2_norm_x, omega_uv, blocked
    )

    v, l2_norm_y = solve(
        v, v_star, A_p, A_e, A_w, A_n, A_s,
        source_y, alpha_uv, epsilon_uv, max_inner_iteration_uv, l2_norm_y, omega_uv, blocked
    )

    u_face, v_face = face_velocity(u, v, u_face, v_face, p, A_p, alpha_uv, blocked)

    Ap_p, Ap_e, Ap_w, Ap_n, Ap_s, source_p = pressure_correction_link_coefficients(
        u, u_face, v_face, Ap_p, Ap_e, Ap_w, Ap_n, Ap_s,
        source_p, A_p, A_e, A_w, A_n, A_s, alpha_uv, blocked
    )

    p_prime, l2_norm_p = solve(
        p_prime, p_prime, Ap_p, Ap_e, Ap_w, Ap_n, Ap_s,
        source_p, dummy_alpha_p, epsilon_p, max_inner_iteration_p, l2_norm_p, omega_p, blocked
    )

    p_prime = fix_pressure_reference(p_prime, blocked)

    p_star = correct_pressure(p_star, p, p_prime, alpha_p, blocked)

    u_star, v_star = correct_cell_center_velocity(u, v, u_star, v_star, p_prime, A_p, alpha_uv, blocked)

    u_face, v_face = correct_face_velocity(u_face, v_face, p_prime, A_p, alpha_uv, blocked)

    p = np.copy(p_star)

    iter_end = time.time()


    print(f"Iter {n:4d}: l2_u = {l2_norm_x:.3e}, l2_v = {l2_norm_y:.3e}, "
          f"l2_p = {l2_norm_p:.3e}, time = {iter_end - iter_start:.3f}s")

    max_div = np.abs(source_p[1:n_y + 1, 1:n_x + 1][~blocked[1:n_y + 1, 1:n_x + 1]]).max()
    max_u = np.abs(u[1:n_y + 1, 1:n_x + 1][~blocked[1:n_y + 1, 1:n_x + 1]]).max()
    max_v = np.abs(v[1:n_y + 1, 1:n_x + 1][~blocked[1:n_y + 1, 1:n_x + 1]]).max()
    max_p = np.abs(p[1:n_y + 1, 1:n_x + 1][~blocked[1:n_y + 1, 1:n_x + 1]]).max()

    print(f"  Max div: {max_div:.3e}, Max u: {max_u:.3e}, "
          f"Max v: {max_v:.3e}, Max p: {max_p:.3e}")

    # Check for divergence
    if max_u > 10 or max_v > 10 or max_p > 1000:
        print("SOLUTION DIVERGING - stopping")
        break

    if (l2_norm_x < 1e-5) and (l2_norm_y < 1e-5) and (l2_norm_p < 1e-5):
        max_div_final = np.abs(source_p[1:n_y + 1, 1:n_x + 1][~blocked[1:n_y + 1, 1:n_x + 1]]).max()
        if max_div_final < 1e-6:
            print(f"\nConverged at iteration {n}!")
            print(f"Final residuals: l2_u = {l2_norm_x:.3e}, l2_v = {l2_norm_y:.3e}, "
                  f"l2_p = {l2_norm_p:.3e}")
            print(f"Max divergence: {max_div_final:.3e}")
            break

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
time_end = time.time()
print(f'\nTotal elapsed time: {time_end - time_start:.1f} seconds')
print(f'Average time per iteration: {(time_end - time_start) / n:.3f} seconds')

# Save solution
np.savez(f'solution_cavity_step_{n_x}x{n_y}_Re{Re}.npz',
         u=u_star[1:n_y + 1, 1:n_x + 1],
         v=v_star[1:n_y + 1, 1:n_x + 1],
         p=p_star[1:n_y + 1, 1:n_x + 1],
         x=x[1:n_x + 1],
         y=y[1:n_y + 1],
         blocked=blocked[1:n_y + 1, 1:n_x + 1],
         Re=Re)

post_processing(
    u_star, v_star, p_star,
    X, Y, x, y, blocked,
    Re=Re,
    step_x_fraction=step_x_fraction,
    step_y_fraction=step_y_fraction,
    save=True,
    prefix=f"cavity_step_{n_x}x{n_y}_Re{Re}"
)

