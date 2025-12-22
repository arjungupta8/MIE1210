import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import RectBivariateSpline
import time

def momentum_link_coefficients(u_star, u_face, v_face, p, source_x, source_y,
                               A_p, A_e, A_w, A_n, A_s, blocked):
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

def solve (phi, phi_star, A_p, A_e, A_w, A_n, A_s, source,
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

def face_velocity(u, v, u_face, v_face, p, A_p, alpha_uv, blocked):

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



def pressure_correction_link_coefficients(u_face, v_face,
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


def correct_pressure(p, p_prime, alpha_p, blocked):
    p_star = p + alpha_p * p_prime

    # BCs for ghost cells (same as before)
    p_star[0, 1:n_x + 1] = p_star[1, 1:n_x + 1]
    p_star[1:n_y + 1, 0] = p_star[1:n_y + 1, 1]
    p_star[1:n_y + 1, n_x + 1] = p_star[1:n_y + 1, n_x]
    p_star[n_y + 1, 1:n_x + 1] = p_star[n_y, 1:n_x + 1]

    p_star[0, 0] = (p_star[1, 2] + p_star[0, 1] + p_star[1, 0]) / 3.0
    p_star[0, n_x + 1] = (p_star[0, n_x] + p_star[1, n_x] + p_star[1, n_x + 1]) / 3.0
    p_star[n_y + 1, 0] = (p_star[n_y, 0] + p_star[n_y, 1] + p_star[n_y + 1, 1]) / 3.0
    p_star[n_y + 1, n_x + 1] = (p_star[n_y, n_x + 1] + p_star[n_y + 1, n_x] + p_star[n_y, n_x]
                               ) / 3.0

    return p_star


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





# Main creation

n_x = 320
n_y = 320
dx = 1.0 / n_x
dy = 1.0 / n_y
Re = 200


alpha_uv = 0.05
alpha_p = 0.004
dummy_alpha_p = 1.0 # Used in GS-SOR for alpha_p. Unrelaxed in GS part of solve()
omega_uv = 0.8
omega_p = 0.8
epsilon_uv = 1e-5
epsilon_p = 1e-6
max_inner_iteration_uv = 25
max_inner_iteration_p = 125
max_outer_iteration = 2000

l2_norm_x = 0.0
l2_norm_y = 0.0
l2_norm_p = 0.0

step_x_fraction = 0.25  # Step extends 25% in x-direction
step_y_fraction = 0.25  # Step extends 25% in y-direction
step_x_cells = int(n_x * step_x_fraction)
step_y_cells = int(n_y * step_y_fraction)
# Create a mask for blocked/solid cells
blocked = np.zeros((n_y + 2, n_x + 2), dtype=bool)
# Block bottom-left corner cells for interior nodes only
blocked[n_y - step_y_cells + 1:n_y + 1, 1:step_x_cells + 1] = True

print(f"Initial Guess of Zero")
u = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
v = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
p = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)

u_star = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
v_star = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)
p_star = np.zeros((n_y + 2, n_x + 2), dtype=np.float64)

# Lid boundary condition
u[0, 1:n_x + 1] = 1.0
u_star[0, 1:n_x + 1] = 1.0

# arrays that need to be initialized no matter what
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

time_start = time.time()

# SIMPLE Loop - steps from Section 1.10
for n in range(1, max_outer_iteration + 1):
    iter_start = time.time()
    l2_norm_p_prev = l2_norm_p if n > 1 else 1.0

    # Steps 2: Convective coefficients and Source Terms
    A_p, A_e, A_w, A_n, A_s, source_x, source_y = momentum_link_coefficients(
        u_star, u_face, v_face, p, source_x, source_y, A_p, A_e, A_w, A_n, A_s, blocked
        )

    # Step 3: Solve discretized momentum equations
    u, l2_norm_x = solve(
        u, u_star, A_p, A_e, A_w, A_n, A_s, source_x, alpha_uv, epsilon_uv,
        max_inner_iteration_uv, l2_norm_x, omega_uv, blocked
        )
    v, l2_norm_y = solve(
        v, v_star, A_p, A_e, A_w, A_n, A_s,
        source_y, alpha_uv, epsilon_uv, max_inner_iteration_uv, l2_norm_y, omega_uv, blocked
    )

    # Step 4: Face Velocities
    u_face, v_face = face_velocity(u, v, u_face, v_face, p, A_p, alpha_uv, blocked)

    # Step 5: Convective coefficients and Source Terms for pressure equations
    Ap_p, Ap_e, Ap_w, Ap_n, Ap_s, source_p = pressure_correction_link_coefficients(
        u_face, v_face, Ap_p, Ap_e, Ap_w, Ap_n, Ap_s,
        source_p, A_p, A_e, A_w, A_n, A_s, alpha_uv, blocked
    )

    # Step 6: Solve the discretized pressure correction equations
    p_prime, l2_norm_p = solve(
        p_prime, p_prime, Ap_p, Ap_e, Ap_w, Ap_n, Ap_s,
        source_p, dummy_alpha_p, epsilon_p, max_inner_iteration_p, l2_norm_p, omega_p, blocked
    )

    # Step 7: Pressure Reference established - skipped. Made convergence worse

    # Step 8: Correct pressure and velocity fields
    p_star = correct_pressure(p, p_prime, alpha_p, blocked)
    u_star, v_star = correct_cell_center_velocity(u, v, u_star, v_star, p_prime, A_p, alpha_uv, blocked)

    # Step 9: Correct face velocities
    u_face, v_face = correct_face_velocity(u_face, v_face, p_prime, A_p, alpha_uv, blocked)

    # Step 10: Pressure Update + Convergence/Divergence Check
    p = np.copy(p_star)
    max_div = np.abs(source_p[1:n_y + 1, 1:n_x + 1]).max()
    max_u = np.abs(u[1:n_y + 1, 1:n_x + 1]).max()
    max_v = np.abs(v[1:n_y + 1, 1:n_x + 1]).max()
    max_p = np.abs(p[1:n_y + 1, 1:n_x + 1]).max()

    # Outputs to console for my own troubleshooting and to see progress lol
    iter_end = time.time()
    print(f"Iter {n:4d}: l2_u = {l2_norm_x: .3e}, l2_v = {l2_norm_y: .3e}, l2_p = {l2_norm_p: .3e}, iter_time = {iter_end - iter_start:.3e}")
    print(f"  Max div: {max_div:.3e}, Max u: {max_u:.3e}, Max v: {max_v:.3e}, Max p: {max_p:.3e}")

    # Check for divergence
    if max_u > 10 or max_v > 10 or max_p > 1000:
        print("SOLUTION DIVERGING - stopping")
        break

    if (l2_norm_x < 1e-4) and (l2_norm_y < 1e-4) and (l2_norm_p < 1e-4) and (max_div < 1e-5):
        print("Converged!")
        break

time_end = time.time()
print(f'total elapsed time in seconds: {time_end - time_start:.1f}')
