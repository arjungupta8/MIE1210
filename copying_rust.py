import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def ix(i, j, width):
    """Index mapping from 2D to 1D array."""
    return i + j * width


def tridiagonal_solver(a, b, c, d, n_eq):
    """
    Thomas algorithm for tridiagonal systems.
    a: lower diagonal (indexed 1..n, a[0] unused)
    b: main diagonal (0..n)
    c: upper diagonal (0..n-1)
    d: right-hand side (0..n)
    """
    c = c.copy()
    d = d.copy()

    c[0] /= b[0]
    d[0] /= b[0]

    for i in range(1, n_eq - 1):
        m = 1.0 / (b[i] - a[i] * c[i - 1])
        c[i] *= m
        d[i] = (d[i] - a[i] * d[i - 1]) * m

    d[n_eq - 1] = (d[n_eq - 1] - a[n_eq - 1] * d[n_eq - 2]) / (b[n_eq - 1] - a[n_eq - 1] * c[n_eq - 2])

    for i in range(n_eq - 2, -1, -1):
        d[i] -= c[i] * d[i + 1]

    return d


class LidDrivenCavity:
    def __init__(self, nx=80, ny=80, re=100.0, relax_uv=0.8, relax_p=0.1, damping=0.2):
        self.nx = nx
        self.ny = ny
        self.re = re
        self.nu = 1.0 / re
        self.rho = 1.0
        self.relax_uv = relax_uv
        self.relax_p = relax_p
        self.damping = damping

        self.width = 1.0
        self.height = 1.0
        self.dx = self.width / nx
        self.dy = self.height / ny

        # Initialize fields
        self.u = np.zeros(ny * nx)
        self.v = np.zeros(ny * nx)
        self.p = np.zeros(ny * nx)
        self.pc = np.zeros(ny * nx)

        # Link coefficients for momentum equations
        self.links_e = np.zeros(ny * nx)
        self.links_w = np.zeros(ny * nx)
        self.links_n = np.zeros(ny * nx)
        self.links_s = np.zeros(ny * nx)
        self.a_0 = np.zeros(ny * nx)

        # Link coefficients for pressure correction
        self.plinks_e = np.zeros(ny * nx)
        self.plinks_w = np.zeros(ny * nx)
        self.plinks_n = np.zeros(ny * nx)
        self.plinks_s = np.zeros(ny * nx)
        self.a_p0 = np.zeros(ny * nx)

        # Source terms
        self.source_x = np.zeros(ny * nx)
        self.source_y = np.zeros(ny * nx)
        self.source_p = np.zeros(ny * nx)

        # Face velocities
        self.u_e = np.zeros(ny * nx)
        self.u_w = np.zeros(ny * nx)
        self.v_n = np.zeros(ny * nx)
        self.v_s = np.zeros(ny * nx)

        # Residuals
        self.res_u = []
        self.res_v = []
        self.res_p = []

    def get_links_momentum(self):
        """
        STEP 1: Calculate momentum equation coefficients.
        File: get_links_momentum.rs

        This implements the discretized momentum equations using:
        - Hybrid differencing (upwind for convection)
        - Central differencing for diffusion
        - Deferred correction approach
        """
        n = self.nx

        # Diffusion coefficients (always positive)
        d_e = self.nu * self.dy / self.dx
        d_w = self.nu * self.dy / self.dx
        d_n = self.nu * self.dx / self.dy
        d_s = self.nu * self.dx / self.dy

        # Interior cells
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                idx = ix(i, j, n)

                # East coefficient: convection + diffusion
                self.links_e[idx] = -max(-self.u_e[idx] * self.dy, 0.0) - d_e
                # West coefficient
                self.links_w[idx] = -max(self.u_w[idx] * self.dy, 0.0) - d_w
                # North coefficient
                self.links_n[idx] = -max(-self.v_n[idx] * self.dx, 0.0) - d_n
                # South coefficient
                self.links_s[idx] = -max(self.v_s[idx] * self.dx, 0.0) - d_s

                # Central coefficient (sum of absolute neighbor coefficients)
                self.a_0[idx] = (max(self.u_e[idx] * self.dy, 0.0) + d_e +
                                 max(-self.u_w[idx] * self.dy, 0.0) + d_w +
                                 max(self.v_n[idx] * self.dx, 0.0) + d_n +
                                 max(-self.v_s[idx] * self.dx, 0.0) + d_s)

                # Source term from pressure gradient
                self.source_x[idx] = 0.5 * (self.p[ix(i - 1, j, n)] - self.p[ix(i + 1, j, n)]) * self.dy
                self.source_y[idx] = 0.5 * (self.p[ix(i, j - 1, n)] - self.p[ix(i, j + 1, n)]) * self.dx

        # Left wall (i=0)
        for j in range(1, self.ny - 1):
            idx = ix(0, j, n)
            self.links_e[idx] = -max(-self.u_e[idx] * self.dy, 0.0) - d_e - d_w / 3.0
            self.links_w[idx] = 0.0
            self.links_n[idx] = -max(-self.v_n[idx] * self.dx, 0.0) - d_n
            self.links_s[idx] = -max(self.v_s[idx] * self.dx, 0.0) - d_s

            self.a_0[idx] = (max(self.u_e[idx] * self.dy, 0.0) + d_e +
                             max(self.v_n[idx] * self.dx, 0.0) + d_n +
                             3.0 * d_w + max(-self.v_s[idx] * self.dx, 0.0) + d_s)

            self.source_x[idx] = 0.5 * (self.p[ix(0, j, n)] - self.p[ix(1, j, n)]) * self.dy
            self.source_y[idx] = 0.5 * (self.p[ix(0, j - 1, n)] - self.p[ix(0, j + 1, n)]) * self.dx

        # Right wall (i=nx-1)
        for j in range(1, self.ny - 1):
            idx = ix(self.nx - 1, j, n)
            self.links_e[idx] = 0.0
            self.links_w[idx] = -max(self.u_w[idx] * self.dy, 0.0) - d_w - d_e / 3.0
            self.links_n[idx] = -max(-self.v_n[idx] * self.dx, 0.0) - d_n
            self.links_s[idx] = -max(self.v_s[idx] * self.dx, 0.0) - d_s

            self.a_0[idx] = (max(-self.u_w[idx] * self.dy, 0.0) + d_w + 3.0 * d_e +
                             max(self.v_n[idx] * self.dx, 0.0) + d_n +
                             max(-self.v_s[idx] * self.dx, 0.0) + d_s)

            self.source_x[idx] = 0.5 * (self.p[ix(self.nx - 1 - 1, j, n)] - self.p[ix(self.nx - 1, j, n)]) * self.dy
            self.source_y[idx] = 0.5 * (self.p[ix(self.nx - 1, j - 1, n)] - self.p[ix(self.nx - 1, j + 1, n)]) * self.dx

        # Top wall (j=ny-1) - lid moving with u=1
        for i in range(1, self.nx - 1):
            idx = ix(i, self.ny - 1, n)
            self.links_e[idx] = -max(-self.u_e[idx] * self.dy, 0.0) - d_e
            self.links_w[idx] = -max(self.u_w[idx] * self.dy, 0.0) - d_w
            self.links_n[idx] = 0.0
            self.links_s[idx] = -max(self.v_s[idx] * self.dx, 0.0) - d_s - d_n / 3.0

            self.a_0[idx] = (max(self.u_e[idx] * self.dy, 0.0) + d_e +
                             max(-self.u_w[idx] * self.dy, 0.0) + d_w +
                             max(-self.v_s[idx] * self.dx, 0.0) + d_s + 3.0 * d_n)

            # Top wall has u=1 boundary condition, adding source term
            self.source_x[idx] = 0.5 * (self.p[ix(i - 1, self.ny - 1, n)] - self.p[
                ix(i + 1, self.ny - 1, n)]) * self.dy + d_n * 8.0 / 3.0 * 1.0
            self.source_y[idx] = 0.5 * (self.p[ix(i, self.ny - 1 - 1, n)] - self.p[ix(i, self.ny - 1, n)]) * self.dx

        # Bottom wall (j=0)
        for i in range(1, self.nx - 1):
            idx = ix(i, 0, n)
            self.links_e[idx] = -max(-self.u_e[idx] * self.dy, 0.0) - d_e
            self.links_w[idx] = -max(self.u_w[idx] * self.dy, 0.0) - d_w
            self.links_n[idx] = -max(-self.v_n[idx] * self.dx, 0.0) - d_n - d_s / 3.0
            self.links_s[idx] = 0.0

            self.a_0[idx] = (max(self.u_e[idx] * self.dy, 0.0) + d_e +
                             max(-self.u_w[idx] * self.dy, 0.0) + d_w +
                             max(self.v_n[idx] * self.dx, 0.0) + d_n + 3.0 * d_s)

            self.source_x[idx] = 0.5 * (self.p[ix(i - 1, 0, n)] - self.p[ix(i + 1, 0, n)]) * self.dy
            self.source_y[idx] = 0.5 * (self.p[ix(i, 0, n)] - self.p[ix(i, 1, n)]) * self.dx

        # Corners - handle specially
        self._set_corner_momentum(0, self.ny - 1, d_e, d_w, d_n, d_s)
        self._set_corner_momentum(0, 0, d_e, d_w, d_n, d_s)
        self._set_corner_momentum(self.nx - 1, self.ny - 1, d_e, d_w, d_n, d_s)
        self._set_corner_momentum(self.nx - 1, 0, d_e, d_w, d_n, d_s)

    def _set_corner_momentum(self, i, j, d_e, d_w, d_n, d_s):
        """Set momentum coefficients for corner cells."""
        n = self.nx
        idx = ix(i, j, n)

        if i == 0 and j == self.ny - 1:  # Top-left
            self.links_e[idx] = -max(-self.u_e[idx] * self.dy, 0.0) - d_e - d_w / 3.0
            self.links_w[idx] = 0.0
            self.links_n[idx] = 0.0
            self.links_s[idx] = -max(self.v_s[idx] * self.dx, 0.0) - d_s - d_n / 3.0
            self.a_0[idx] = max(self.u_e[idx] * self.dy, 0.0) + d_e + max(-self.v_s[idx] * self.dx,
                                                                          0.0) + d_s + 3.0 * d_n + 3.0 * d_w
            self.source_x[idx] = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i + 1, j, n)]) * self.dy
            self.source_y[idx] = 0.5 * (self.p[ix(i, j - 1, n)] - self.p[ix(i, j, n)]) * self.dx
        elif i == 0 and j == 0:  # Bottom-left
            self.links_e[idx] = -max(-self.u_e[idx] * self.dy, 0.0) - d_e - d_w / 3.0
            self.links_w[idx] = 0.0
            self.links_n[idx] = -max(-self.v_n[idx] * self.dx, 0.0) - d_n - d_s / 3.0
            self.links_s[idx] = 0.0
            self.a_0[idx] = max(self.u_e[idx] * self.dy, 0.0) + d_e + max(self.v_n[idx] * self.dx,
                                                                          0.0) + d_n + 3.0 * d_s + 3.0 * d_w
            self.source_x[idx] = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i + 1, j, n)]) * self.dy
            self.source_y[idx] = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i, j + 1, n)]) * self.dx
        elif i == self.nx - 1 and j == self.ny - 1:  # Top-right
            self.links_e[idx] = 0.0
            self.links_w[idx] = -max(self.u_w[idx] * self.dy, 0.0) - d_w - d_e / 3.0
            self.links_n[idx] = 0.0
            self.links_s[idx] = -max(self.v_s[idx] * self.dx, 0.0) - d_s - d_n / 3.0
            self.a_0[idx] = max(-self.u_w[idx] * self.dy, 0.0) + d_w + max(-self.v_s[idx] * self.dx,
                                                                           0.0) + d_s + 3.0 * d_n + 3.0 * d_e
            self.source_x[idx] = 0.5 * (self.p[ix(i - 1, j, n)] - self.p[ix(i, j, n)]) * self.dy
            self.source_y[idx] = 0.5 * (self.p[ix(i, j - 1, n)] - self.p[ix(i, j, n)]) * self.dx
        elif i == self.nx - 1 and j == 0:  # Bottom-right
            self.links_e[idx] = 0.0
            self.links_w[idx] = -max(self.u_w[idx] * self.dy, 0.0) - d_w - d_e / 3.0
            self.links_n[idx] = -max(-self.v_n[idx] * self.dx, 0.0) - d_n - d_s / 3.0
            self.links_s[idx] = 0.0
            self.a_0[idx] = max(-self.u_w[idx] * self.dy, 0.0) + d_w + max(self.v_n[idx] * self.dx,
                                                                           0.0) + 3.0 * d_e + d_n + 3.0 * d_s
            self.source_x[idx] = 0.5 * (self.p[ix(i - 1, j, n)] - self.p[ix(i, j, n)]) * self.dy
            self.source_y[idx] = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i, j + 1, n)]) * self.dx

    def solver_correction(self, x, sources, n_iter=2):
        """
        STEP 2: Solve momentum equations with correction.
        File: solver_correction.rs

        Uses line-by-line ADI (Alternating Direction Implicit) method:
        1. Sweep in X-direction (solve tridiagonal systems for each row)
        2. Sweep in Y-direction (solve tridiagonal systems for each column)

        The 'damping' parameter adds inertial under-relaxation for stability.
        """
        n = self.nx

        for _ in range(n_iter):
            # X-direction sweep
            for j in range(self.ny):
                diagonal = np.zeros(self.nx)
                ax = np.zeros(self.nx)
                cx = np.zeros(self.nx - 1)
                rhs = np.zeros(self.nx)

                for i in range(self.nx):
                    idx = ix(i, j, n)
                    diagonal[i] = (1.0 + self.damping) * self.a_0[idx]

                    if i < self.nx - 1:
                        cx[i] = self.links_e[idx]
                    if i > 0:
                        ax[i] = self.links_w[idx]

                    rhs[i] = sources[idx]

                    # Add contributions from Y-direction neighbors
                    if j > 0:
                        rhs[i] -= self.links_s[idx] * x[ix(i, j - 1, n)]
                    if j < self.ny - 1:
                        rhs[i] -= self.links_n[idx] * x[ix(i, j + 1, n)]

                    # Add contributions from X-direction neighbors already solved
                    if i > 0:
                        rhs[i] -= self.links_e[ix(i - 1, j, n)] * x[ix(i - 1, j, n)]
                    if i < self.nx - 1:
                        rhs[i] -= self.links_w[ix(i + 1, j, n)] * x[ix(i + 1, j, n)]

                    # Damping term
                    rhs[i] -= self.a_0[idx] * x[idx]

                # Solve tridiagonal system
                sol = tridiagonal_solver(ax, diagonal, cx, rhs, self.nx)

                # Update solution (ADDITIVE correction)
                for i in range(self.nx):
                    x[ix(i, j, n)] += sol[i]

            # Y-direction sweep
            for i in range(self.nx):
                diagonal = np.zeros(self.ny)
                ax = np.zeros(self.ny)
                cx = np.zeros(self.ny - 1)
                rhs = np.zeros(self.ny)

                for j in range(self.ny):
                    idx = ix(i, j, n)
                    diagonal[j] = (1.0 + self.damping) * self.a_0[idx]

                    if j < self.ny - 1:
                        cx[j] = self.links_n[idx]
                    if j > 0:
                        ax[j] = self.links_s[idx]

                    rhs[j] = sources[idx]

                    # Add contributions from X-direction neighbors
                    if i > 0:
                        rhs[j] -= self.links_w[idx] * x[ix(i - 1, j, n)]
                    if i < self.nx - 1:
                        rhs[j] -= self.links_e[idx] * x[ix(i + 1, j, n)]

                    # Add contributions from Y-direction neighbors already solved
                    if j > 0:
                        rhs[j] -= self.links_n[ix(i, j - 1, n)] * x[ix(i, j - 1, n)]
                    if j < self.ny - 1:
                        rhs[j] -= self.links_s[ix(i, j + 1, n)] * x[ix(i, j + 1, n)]

                    # Damping term
                    rhs[j] -= self.a_0[idx] * x[idx]

                # Solve tridiagonal system
                sol = tridiagonal_solver(ax, diagonal, cx, rhs, self.ny)

                # Update solution (ADDITIVE correction)
                for j in range(self.ny):
                    x[ix(i, j, n)] += sol[j]

    def get_face_velocities(self):
        """
        STEP 3: Calculate face velocities using Rhie-Chow interpolation.
        File: face_velocity.rs

        This prevents checkerboard pressure oscillations by using
        pressure-weighted interpolation (PWIM) for face velocities.
        """
        n = self.nx

        def pwim(vel, a_0, dpdx_0, dpdx_other, dpdx_face, idx, idx_other, d):
            """Pressure-Weighted Interpolation Method"""
            return (0.5 * (vel[idx] + vel[idx_other]) +
                    0.5 * (dpdx_0 / a_0[idx] + dpdx_other / a_0[idx_other] -
                           (1.0 / a_0[idx] + 1.0 / a_0[idx_other]) * dpdx_face) * d)

        # Interior u-faces
        for j in range(self.ny):
            for i in range(2, self.nx - 2):
                idx = ix(i, j, n)
                dpdx_0 = 0.5 * (self.p[ix(i + 1, j, n)] - self.p[ix(i - 1, j, n)])
                dpdx_e = 0.5 * (self.p[ix(i + 2, j, n)] - self.p[ix(i, j, n)])
                dpdx_w = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i - 2, j, n)])
                dpdx_eface = self.p[ix(i + 1, j, n)] - self.p[ix(i, j, n)]
                dpdx_wface = self.p[ix(i, j, n)] - self.p[ix(i - 1, j, n)]

                self.u_e[idx] = pwim(self.u, self.a_0, dpdx_0, dpdx_e, dpdx_eface, idx, ix(i + 1, j, n), self.dy)
                self.u_w[idx] = pwim(self.u, self.a_0, dpdx_0, dpdx_w, dpdx_wface, idx, ix(i - 1, j, n), self.dy)

        # Boundary u-faces (similar but with modified gradients)
        for j in range(self.ny):
            i = 1
            idx = ix(i, j, n)
            dpdx_0 = 0.5 * (self.p[ix(i + 1, j, n)] - self.p[ix(i - 1, j, n)])
            dpdx_e = 0.5 * (self.p[ix(i + 2, j, n)] - self.p[ix(i, j, n)])
            dpdx_w = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i - 1, j, n)])
            dpdx_eface = self.p[ix(i + 1, j, n)] - self.p[ix(i, j, n)]
            dpdx_wface = self.p[ix(i, j, n)] - self.p[ix(i - 1, j, n)]

            self.u_e[idx] = pwim(self.u, self.a_0, dpdx_0, dpdx_e, dpdx_eface, idx, ix(i + 1, j, n), self.dy)
            self.u_w[idx] = pwim(self.u, self.a_0, dpdx_0, dpdx_w, dpdx_wface, idx, ix(i - 1, j, n), self.dy)

            i = 0
            idx = ix(i, j, n)
            dpdx_0 = 0.5 * (self.p[ix(i + 1, j, n)] - self.p[ix(i, j, n)])
            dpdx_e = 0.5 * (self.p[ix(i + 2, j, n)] - self.p[ix(i, j, n)])
            dpdx_eface = self.p[ix(i + 1, j, n)] - self.p[ix(i, j, n)]
            self.u_e[idx] = pwim(self.u, self.a_0, dpdx_0, dpdx_e, dpdx_eface, idx, ix(i + 1, j, n), self.dy)
            self.u_w[idx] = 0.0

            i = self.nx - 2
            idx = ix(i, j, n)
            dpdx_0 = 0.5 * (self.p[ix(i + 1, j, n)] - self.p[ix(i - 1, j, n)])
            dpdx_e = 0.5 * (self.p[ix(i + 1, j, n)] - self.p[ix(i, j, n)])
            dpdx_w = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i - 2, j, n)])
            dpdx_eface = self.p[ix(i + 1, j, n)] - self.p[ix(i, j, n)]
            dpdx_wface = self.p[ix(i, j, n)] - self.p[ix(i - 1, j, n)]
            self.u_e[idx] = pwim(self.u, self.a_0, dpdx_0, dpdx_e, dpdx_eface, idx, ix(i + 1, j, n), self.dy)
            self.u_w[idx] = pwim(self.u, self.a_0, dpdx_0, dpdx_w, dpdx_wface, idx, ix(i - 1, j, n), self.dy)

            i = self.nx - 1
            idx = ix(i, j, n)
            dpdx_0 = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i - 1, j, n)])
            dpdx_w = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i - 2, j, n)])
            dpdx_wface = self.p[ix(i, j, n)] - self.p[ix(i - 1, j, n)]
            self.u_e[idx] = 0.0
            self.u_w[idx] = pwim(self.u, self.a_0, dpdx_0, dpdx_w, dpdx_wface, idx, ix(i - 1, j, n), self.dy)

        # Interior v-faces
        for j in range(2, self.ny - 2):
            for i in range(self.nx):
                idx = ix(i, j, n)
                dpdy_0 = 0.5 * (self.p[ix(i, j + 1, n)] - self.p[ix(i, j - 1, n)])
                dpdy_n = 0.5 * (self.p[ix(i, j + 2, n)] - self.p[ix(i, j, n)])
                dpdy_s = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i, j - 2, n)])
                dpdy_nface = self.p[ix(i, j + 1, n)] - self.p[ix(i, j, n)]
                dpdy_sface = self.p[ix(i, j, n)] - self.p[ix(i, j - 1, n)]

                self.v_n[idx] = pwim(self.v, self.a_0, dpdy_0, dpdy_n, dpdy_nface, idx, ix(i, j + 1, n), self.dx)
                self.v_s[idx] = pwim(self.v, self.a_0, dpdy_0, dpdy_s, dpdy_sface, idx, ix(i, j - 1, n), self.dx)

        # Boundary v-faces
        for i in range(self.nx):
            j = 1
            idx = ix(i, j, n)
            dpdy_0 = 0.5 * (self.p[ix(i, j + 1, n)] - self.p[ix(i, j - 1, n)])
            dpdy_n = 0.5 * (self.p[ix(i, j + 2, n)] - self.p[ix(i, j, n)])
            dpdy_s = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i, j - 1, n)])
            dpdy_nface = self.p[ix(i, j + 1, n)] - self.p[ix(i, j, n)]
            dpdy_sface = self.p[ix(i, j, n)] - self.p[ix(i, j - 1, n)]
            self.v_n[idx] = pwim(self.v, self.a_0, dpdy_0, dpdy_n, dpdy_nface, idx, ix(i, j + 1, n), self.dx)
            self.v_s[idx] = pwim(self.v, self.a_0, dpdy_0, dpdy_s, dpdy_sface, idx, ix(i, j - 1, n), self.dx)

            j = 0
            idx = ix(i, j, n)
            dpdy_0 = 0.5 * (self.p[ix(i, j + 1, n)] - self.p[ix(i, j, n)])
            dpdy_n = 0.5 * (self.p[ix(i, j + 2, n)] - self.p[ix(i, j, n)])
            dpdy_nface = self.p[ix(i, j + 1, n)] - self.p[ix(i, j, n)]
            self.v_n[idx] = pwim(self.v, self.a_0, dpdy_0, dpdy_n, dpdy_nface, idx, ix(i, j + 1, n), self.dx)
            self.v_s[idx] = 0.0

            j = self.ny - 2
            idx = ix(i, j, n)
            dpdy_0 = 0.5 * (self.p[ix(i, j + 1, n)] - self.p[ix(i, j - 1, n)])
            dpdy_n = 0.5 * (self.p[ix(i, j + 1, n)] - self.p[ix(i, j, n)])
            dpdy_s = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i, j - 2, n)])
            dpdy_nface = self.p[ix(i, j + 1, n)] - self.p[ix(i, j, n)]
            dpdy_sface = self.p[ix(i, j, n)] - self.p[ix(i, j - 1, n)]
            self.v_n[idx] = pwim(self.v, self.a_0, dpdy_0, dpdy_n, dpdy_nface, idx, ix(i, j + 1, n), self.dx)
            self.v_s[idx] = pwim(self.v, self.a_0, dpdy_0, dpdy_s, dpdy_sface, idx, ix(i, j - 1, n), self.dx)

            j = self.ny - 1
            idx = ix(i, j, n)
            dpdy_0 = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i, j - 1, n)])
            dpdy_s = 0.5 * (self.p[ix(i, j, n)] - self.p[ix(i, j - 2, n)])
            dpdy_sface = self.p[ix(i, j, n)] - self.p[ix(i, j - 1, n)]
            self.v_n[idx] = 0.0
            self.v_s[idx] = pwim(self.v, self.a_0, dpdy_0, dpdy_s, dpdy_sface, idx, ix(i, j - 1, n), self.dx)

    def get_links_pressure_correction(self):
        """
        STEP 4: Calculate pressure correction equation coefficients.
        File: get_links_pressure_correction.rs

        Derives from continuity equation, creating a Poisson-like equation
        for pressure correction.
        """
        n = self.nx

        # Interior cells
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                idx = ix(i, j, n)
                a_0 = self.a_0[idx]
                a_0_e = self.a_0[ix(i + 1, j, n)]
                a_0_w = self.a_0[ix(i - 1, j, n)]
                a_0_n = self.a_0[ix(i, j + 1, n)]
                a_0_s = self.a_0[ix(i, j - 1, n)]

                self.plinks_e[idx] = -0.5 * self.dy * self.dy * (1.0 / a_0_e + 1.0 / a_0)
                self.plinks_w[idx] = -0.5 * self.dy * self.dy * (1.0 / a_0_w + 1.0 / a_0)
                self.plinks_n[idx] = -0.5 * self.dx * self.dx * (1.0 / a_0_n + 1.0 / a_0)
                self.plinks_s[idx] = -0.5 * self.dx * self.dx * (1.0 / a_0_s + 1.0 / a_0)

                self.a_p0[idx] = -self.plinks_e[idx] - self.plinks_w[idx] - self.plinks_n[idx] - self.plinks_s[idx]

                # Source is mass imbalance
                mdot = -(self.u_e[idx] - self.u_w[idx]) * self.dy - (self.v_n[idx] - self.v_s[idx]) * self.dx
                self.source_p[idx] = mdot

        # Boundaries
        self._set_all_pressure_boundaries()

    def _set_all_pressure_boundaries(self):
        """Set pressure correction coefficients for all boundaries."""
        n = self.nx

        # Left wall (i=0)
        for j in range(1, self.ny - 1):
            idx = ix(0, j, n)
            a_0 = self.a_0[idx]
            a_0_e = self.a_0[ix(1, j, n)]
            a_0_n = self.a_0[ix(0, j + 1, n)]
            a_0_s = self.a_0[ix(0, j - 1, n)]

            self.plinks_e[idx] = -0.5 * self.dy * self.dy * (1.0 / a_0_e + 1.0 / a_0)
            self.plinks_w[idx] = 0.0
            self.plinks_n[idx] = -0.5 * self.dx * self.dx * (1.0 / a_0_n + 1.0 / a_0)
            self.plinks_s[idx] = -0.5 * self.dx * self.dx * (1.0 / a_0_s + 1.0 / a_0)
            self.a_p0[idx] = -self.plinks_e[idx] - self.plinks_n[idx] - self.plinks_s[idx]

            mdot = -(self.u_e[idx] - self.u_w[idx]) * self.dy - (self.v_n[idx] - self.v_s[idx]) * self.dx
            self.source_p[idx] = mdot

        # Right wall (i=nx-1)
        for j in range(1, self.ny - 1):
            idx = ix(self.nx - 1, j, n)
            a_0 = self.a_0[idx]
            a_0_w = self.a_0[ix(self.nx - 2, j, n)]
            a_0_n = self.a_0[ix(self.nx - 1, j + 1, n)]
            a_0_s = self.a_0[ix(self.nx - 1, j - 1, n)]

            self.plinks_e[idx] = 0.0
            self.plinks_w[idx] = -0.5 * self.dy * self.dy * (1.0 / a_0_w + 1.0 / a_0)
            self.plinks_n[idx] = -0.5 * self.dx * self.dx * (1.0 / a_0_n + 1.0 / a_0)
            self.plinks_s[idx] = -0.5 * self.dx * self.dx * (1.0 / a_0_s + 1.0 / a_0)
            self.a_p0[idx] = -self.plinks_w[idx] - self.plinks_n[idx] - self.plinks_s[idx]

            mdot = -(self.u_e[idx] - self.u_w[idx]) * self.dy - (self.v_n[idx] - self.v_s[idx]) * self.dx
            self.source_p[idx] = mdot

        # Top wall (j=ny-1)
        for i in range(1, self.nx - 1):
            idx = ix(i, self.ny - 1, n)
            a_0 = self.a_0[idx]
            a_0_e = self.a_0[ix(i + 1, self.ny - 1, n)]
            a_0_w = self.a_0[ix(i - 1, self.ny - 1, n)]
            a_0_s = self.a_0[ix(i, self.ny - 2, n)]

            self.plinks_e[idx] = -0.5 * self.dy * self.dy * (1.0 / a_0_e + 1.0 / a_0)
            self.plinks_w[idx] = -0.5 * self.dy * self.dy * (1.0 / a_0_w + 1.0 / a_0)
            self.plinks_n[idx] = 0.0
            self.plinks_s[idx] = -0.5 * self.dx * self.dx * (1.0 / a_0_s + 1.0 / a_0)
            self.a_p0[idx] = -self.plinks_e[idx] - self.plinks_w[idx] - self.plinks_s[idx]

            mdot = -(self.u_e[idx] - self.u_w[idx]) * self.dy - (self.v_n[idx] - self.v_s[idx]) * self.dx
            self.source_p[idx] = mdot

        # Bottom wall (j=0)
        for i in range(1, self.nx - 1):
            idx = ix(i, 0, n)
            a_0 = self.a_0[idx]
            a_0_e = self.a_0[ix(i + 1, 0, n)]
            a_0_w = self.a_0[ix(i - 1, 0, n)]
            a_0_n = self.a_0[ix(i, 1, n)]

            self.plinks_e[idx] = -0.5 * self.dy * self.dy * (1.0 / a_0_e + 1.0 / a_0)
            self.plinks_w[idx] = -0.5 * self.dy * self.dy * (1.0 / a_0_w + 1.0 / a_0)
            self.plinks_n[idx] = -0.5 * self.dx * self.dx * (1.0 / a_0_n + 1.0 / a_0)
            self.plinks_s[idx] = 0.0
            self.a_p0[idx] = -self.plinks_e[idx] - self.plinks_w[idx] - self.plinks_n[idx]

            mdot = -(self.u_e[idx] - self.u_w[idx]) * self.dy - (self.v_n[idx] - self.v_s[idx]) * self.dx
            self.source_p[idx] = mdot

        # Corners
        for i, j in [(0, 0), (0, self.ny - 1), (self.nx - 1, 0), (self.nx - 1, self.ny - 1)]:
            self._set_pressure_corner(i, j)

    def _set_pressure_corner(self, i, j):
        """Set pressure correction for corner cells."""
        n = self.nx
        idx = ix(i, j, n)
        a_0 = self.a_0[idx]

        # Determine which neighbors exist
        has_east = i < self.nx - 1
        has_west = i > 0
        has_north = j < self.ny - 1
        has_south = j > 0

        if has_east:
            a_0_e = self.a_0[ix(i + 1, j, n)]
            self.plinks_e[idx] = -0.5 * self.dy * self.dy * (1.0 / a_0_e + 1.0 / a_0)
        else:
            self.plinks_e[idx] = 0.0

        if has_west:
            a_0_w = self.a_0[ix(i - 1, j, n)]
            self.plinks_w[idx] = -0.5 * self.dy * self.dy * (1.0 / a_0_w + 1.0 / a_0)
        else:
            self.plinks_w[idx] = 0.0

        if has_north:
            a_0_n = self.a_0[ix(i, j + 1, n)]
            self.plinks_n[idx] = -0.5 * self.dx * self.dx * (1.0 / a_0_n + 1.0 / a_0)
        else:
            self.plinks_n[idx] = 0.0

        if has_south:
            a_0_s = self.a_0[ix(i, j - 1, n)]
            self.plinks_s[idx] = -0.5 * self.dx * self.dx * (1.0 / a_0_s + 1.0 / a_0)
        else:
            self.plinks_s[idx] = 0.0

        self.a_p0[idx] = -self.plinks_e[idx] - self.plinks_w[idx] - self.plinks_n[idx] - self.plinks_s[idx]

        mdot = -(self.u_e[idx] - self.u_w[idx]) * self.dy - (self.v_n[idx] - self.v_s[idx]) * self.dx
        self.source_p[idx] = mdot

    def solver(self, x, n_iter=20):
        """
        STEP 5: Solve pressure correction (NO damping).
        File: solver.rs

        Key difference from solver_correction: NO damping term!
        Uses direct replacement, not additive correction.
        """
        n = self.nx

        for _ in range(n_iter):
            # X-direction sweep
            for j in range(self.ny):
                diagonal = np.zeros(self.nx)
                ax = np.zeros(self.nx)
                cx = np.zeros(self.nx - 1)
                rhs = np.zeros(self.nx)

                for i in range(self.nx):
                    idx = ix(i, j, n)
                    diagonal[i] = self.a_p0[idx]

                    if i < self.nx - 1:
                        cx[i] = self.plinks_e[idx]
                    if i > 0:
                        ax[i] = self.plinks_w[idx]

                    rhs[i] = self.source_p[idx]

                    if j > 0:
                        rhs[i] -= self.plinks_s[idx] * x[ix(i, j - 1, n)]
                    if j < self.ny - 1:
                        rhs[i] -= self.plinks_n[idx] * x[ix(i, j + 1, n)]

                sol = tridiagonal_solver(ax, diagonal, cx, rhs, self.nx)
                x[j * n:(j + 1) * n] = sol[:self.nx]  # DIRECT replacement

            # Y-direction sweep
            for i in range(self.nx):
                diagonal = np.zeros(self.ny)
                ax = np.zeros(self.ny)
                cx = np.zeros(self.ny - 1)
                rhs = np.zeros(self.ny)

                for j in range(self.ny):
                    idx = ix(i, j, n)
                    diagonal[j] = self.a_p0[idx]

                    if j < self.ny - 1:
                        cx[j] = self.plinks_n[idx]
                    if j > 0:
                        ax[j] = self.plinks_s[idx]

                    rhs[j] = self.source_p[idx]

                    if i > 0:
                        rhs[j] -= self.plinks_w[idx] * x[ix(i - 1, j, n)]
                    if i < self.nx - 1:
                        rhs[j] -= self.plinks_e[idx] * x[ix(i + 1, j, n)]

                sol = tridiagonal_solver(ax, diagonal, cx, rhs, self.ny)
                for j in range(self.ny):
                    x[ix(i, j, n)] = sol[j]  # DIRECT replacement

    def correct_velocities(self):
        """
        STEP 6: Correct velocities and pressure.
        File: correct_parameters.rs

        Apply pressure correction to update velocities and pressure.
        """
        n = self.nx

        # Correct cell velocities - interior
        for j in range(self.ny):
            for i in range(1, self.nx - 1):
                idx = ix(i, j, n)
                self.u[idx] += self.relax_uv * 0.5 * (self.pc[ix(i - 1, j, n)] - self.pc[ix(i + 1, j, n)]) * self.dy / \
                               self.a_0[idx]

        for i in range(self.nx):
            for j in range(1, self.ny - 1):
                idx = ix(i, j, n)
                self.v[idx] += self.relax_uv * 0.5 * (self.pc[ix(i, j - 1, n)] - self.pc[ix(i, j + 1, n)]) * self.dx / \
                               self.a_0[idx]

        # Correct cell velocities - boundaries
        for j in range(self.ny):
            idx = ix(0, j, n)
            self.u[idx] += self.relax_uv * 0.5 * (self.pc[idx] - self.pc[ix(1, j, n)]) * self.dy / self.a_0[idx]

            idx = ix(self.nx - 1, j, n)
            self.u[idx] += self.relax_uv * 0.5 * (self.pc[ix(self.nx - 2, j, n)] - self.pc[idx]) * self.dy / self.a_0[
                idx]

        for i in range(self.nx):
            idx = ix(i, 0, n)
            self.v[idx] += self.relax_uv * 0.5 * (self.pc[idx] - self.pc[ix(i, 1, n)]) * self.dx / self.a_0[idx]

            idx = ix(i, self.ny - 1, n)
            self.v[idx] += self.relax_uv * 0.5 * (self.pc[ix(i, self.ny - 2, n)] - self.pc[idx]) * self.dx / self.a_0[
                idx]

        # Correct face velocities - interior
        for j in range(self.ny):
            for i in range(1, self.nx - 1):
                idx = ix(i, j, n)
                a_0 = self.a_0[idx]
                a_0e = self.a_0[ix(i + 1, j, n)]
                a_0w = self.a_0[ix(i - 1, j, n)]

                self.u_e[idx] += self.relax_uv * 0.5 * (self.pc[idx] - self.pc[ix(i + 1, j, n)]) * self.dy / (
                            1.0 / a_0 + 1.0 / a_0e)
                self.u_w[idx] += self.relax_uv * 0.5 * (self.pc[ix(i - 1, j, n)] - self.pc[idx]) * self.dy / (
                            1.0 / a_0 + 1.0 / a_0w)

        for i in range(self.nx):
            for j in range(1, self.ny - 1):
                idx = ix(i, j, n)
                a_0 = self.a_0[idx]
                a_0n = self.a_0[ix(i, j + 1, n)]
                a_0s = self.a_0[ix(i, j - 1, n)]

                self.v_n[idx] += self.relax_uv * 0.5 * (self.pc[idx] - self.pc[ix(i, j + 1, n)]) * self.dx / (
                            1.0 / a_0 + 1.0 / a_0n)
                self.v_s[idx] += self.relax_uv * 0.5 * (self.pc[ix(i, j - 1, n)] - self.pc[idx]) * self.dx / (
                            1.0 / a_0 + 1.0 / a_0s)

        # Correct face velocities - boundaries
        for j in range(self.ny):
            idx = ix(0, j, n)
            a_0e = self.a_0[ix(1, j, n)]
            self.u_e[idx] += self.relax_uv * 0.5 * (self.pc[idx] - self.pc[ix(1, j, n)]) * self.dy / (
                        1.0 / self.a_0[idx] + 1.0 / a_0e)

            idx = ix(self.nx - 1, j, n)
            a_0w = self.a_0[ix(self.nx - 2, j, n)]
            self.u_w[idx] += self.relax_uv * 0.5 * (self.pc[ix(self.nx - 2, j, n)] - self.pc[idx]) * self.dy / (
                        1.0 / self.a_0[idx] + 1.0 / a_0w)

        for i in range(self.nx):
            idx = ix(i, 0, n)
            a_0n = self.a_0[ix(i, 1, n)]
            self.v_n[idx] += self.relax_uv * 0.5 * (self.pc[idx] - self.pc[ix(i, 1, n)]) * self.dx / (
                        1.0 / self.a_0[idx] + 1.0 / a_0n)

            idx = ix(i, self.ny - 1, n)
            a_0s = self.a_0[ix(i, self.ny - 2, n)]
            self.v_s[idx] += self.relax_uv * 0.5 * (self.pc[ix(i, self.ny - 2, n)] - self.pc[idx]) * self.dx / (
                        1.0 / self.a_0[idx] + 1.0 / a_0s)

        # Correct pressure
        self.p += self.relax_p * self.pc

    def calculate_residuals(self):
        """
        Calculate residuals for convergence checking.
        File: residuals.rs
        """
        n = self.nx

        # U-velocity residual
        res_u = 0.0
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                idx = ix(i, j, n)
                r = (self.links_e[idx] * self.u[ix(i + 1, j, n)] +
                     self.links_w[idx] * self.u[ix(i - 1, j, n)] +
                     self.links_n[idx] * self.u[ix(i, j + 1, n)] +
                     self.links_s[idx] * self.u[ix(i, j - 1, n)] +
                     self.a_0[idx] * self.u[idx] -
                     self.source_x[idx])
                res_u += r * r
        self.res_u.append(np.sqrt(res_u))

        # V-velocity residual
        res_v = 0.0
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                idx = ix(i, j, n)
                r = (self.links_e[idx] * self.v[ix(i + 1, j, n)] +
                     self.links_w[idx] * self.v[ix(i - 1, j, n)] +
                     self.links_n[idx] * self.v[ix(i, j + 1, n)] +
                     self.links_s[idx] * self.v[ix(i, j - 1, n)] +
                     self.a_0[idx] * self.v[idx] -
                     self.source_y[idx])
                res_v += r * r
        self.res_v.append(np.sqrt(res_v))

        # Pressure residual
        res_p = 0.0
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                idx = ix(i, j, n)
                mdot = -(self.u_e[idx] - self.u_w[idx]) * self.dy - (self.v_n[idx] - self.v_s[idx]) * self.dx
                r = (self.plinks_e[idx] * self.pc[ix(i + 1, j, n)] +
                     self.plinks_w[idx] * self.pc[ix(i - 1, j, n)] +
                     self.plinks_n[idx] * self.pc[ix(i, j + 1, n)] +
                     self.plinks_s[idx] * self.pc[ix(i, j - 1, n)] +
                     self.a_p0[idx] * self.pc[idx] -
                     mdot)
                res_p += r * r
        self.res_p.append(np.sqrt(res_p))

    def iterate(self):
        """
        Main iteration loop - implements SIMPLE algorithm.
        EXACT order from lid_driven_cavity.rs iterate() method!
        """
        # Step 1: Calculate momentum coefficients
        self.get_links_momentum()

        # Step 2: Solve u-momentum and calculate residual
        self.solver_correction(self.u, self.source_x, 2)

        # Step 3: Solve v-momentum and calculate residuals for both
        self.solver_correction(self.v, self.source_y, 2)
        self.calculate_residuals()

        # Step 4: Calculate face velocities using Rhie-Chow interpolation
        self.get_face_velocities()

        # Step 5: Calculate pressure correction coefficients
        self.get_links_pressure_correction()

        # Step 6: Solve pressure correction equation
        self.pc = np.zeros(self.ny * self.nx)
        self.solver(self.pc, 20)

        # Step 7: Correct velocities and pressure
        self.correct_velocities()

        # Print residuals
        print(f"res u: {self.res_u[-1]:.2e}, res v: {self.res_v[-1]:.2e}, res p: {self.res_p[-1]:.2e}")

    def has_converged(self, epsilon=1e-10):
        """Check if solution has converged."""
        if len(self.res_u) == 0:
            return False
        return (self.res_u[-1] < epsilon and
                self.res_v[-1] < epsilon and
                self.res_p[-1] < epsilon)

    def has_diverged(self):
        """Check if solution has diverged."""
        return np.isnan(np.sum(np.abs(self.u)))

    def plot_results(self, iteration):
        """
        Generate visualization plots.
        File: postprocessing.rs
        """
        export_path = Path("./results_lid_driven_cavity/")
        export_path.mkdir(parents=True, exist_ok=True)

        # Reshape data for plotting
        u_2d = self.u.reshape((self.ny, self.nx))
        v_2d = self.v.reshape((self.ny, self.nx))
        p_2d = self.p.reshape((self.ny, self.nx))

        # Create coordinate arrays
        x = np.linspace(0, self.width, self.nx)
        y = np.linspace(0, self.height, self.ny)
        X, Y = np.meshgrid(x, y)

        # Velocity magnitude with streamlines
        vel_mag = np.sqrt(u_2d ** 2 + v_2d ** 2)

        plt.figure(figsize=(6, 6), dpi=100)
        plt.pcolormesh(X, Y, vel_mag, cmap='plasma', shading='auto')
        plt.colorbar(fraction=0.016, pad=0.04)
        plt.streamplot(x, y, u_2d, v_2d, color='w', density=3.0, linewidth=0.5)
        plt.title(f'Velocity magnitude (Re = {self.re:.0f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(export_path / 'velocity_m.png')
        plt.close()

        # U-velocity contour
        plt.figure(figsize=(6, 6), dpi=100)
        plt.contourf(X, Y, u_2d, 30, cmap='jet')
        plt.colorbar(fraction=0.016, pad=0.04)
        plt.title(f'u velocity (Re = {self.re:.0f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(export_path / 'u.png')
        plt.close()

        # V-velocity contour
        plt.figure(figsize=(6, 6), dpi=100)
        plt.contourf(X, Y, v_2d, 30, cmap='jet')
        plt.colorbar(fraction=0.016, pad=0.04)
        plt.title(f'v velocity (Re = {self.re:.0f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(export_path / 'v.png')
        plt.close()

        # Pressure contour
        plt.figure(figsize=(6, 6), dpi=100)
        plt.contourf(X, Y, p_2d, 30, cmap='jet')
        plt.colorbar(fraction=0.016, pad=0.04)
        plt.title(f'Pressure (Re = {self.re:.0f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(export_path / 'p.png')
        plt.close()

        # Comparison with Ghia et al. data
        ghia_u = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
                  -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
                  0.78871, 0.84123, 1.00000]
        ghia_y = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                  0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]

        ghia_v = [0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
                  0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313,
                  -0.08864, -0.07391, -0.05906, 0.0]
        ghia_x = [0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                  0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]

        # Extract centerline profiles
        y_centers = np.linspace(self.dy * 0.5, self.height - self.dy * 0.5, self.ny)
        x_centers = np.linspace(self.dx * 0.5, self.width - self.dx * 0.5, self.nx)

        u_centerline = u_2d[:, self.nx // 2]
        v_centerline = v_2d[self.ny // 2, :]

        plt.figure(figsize=(8, 6), dpi=150)
        plt.plot(y_centers, u_centerline, label='u solution')
        plt.scatter(ghia_y, ghia_u, label='u from Ghia et al.')
        plt.plot(x_centers, v_centerline, label='v solution')
        plt.scatter(ghia_x, ghia_v, label='v from Ghia et al.')
        plt.grid(True)
        plt.legend()
        plt.xlabel('x, y')
        plt.ylabel('u, v')
        plt.savefig(export_path / 'ghia.png')
        plt.close()

        # Residual plot
        if len(self.res_u) > 1:
            iterations = np.arange(1, len(self.res_u) + 1)
            plt.figure(figsize=(8, 6), dpi=150)
            plt.semilogy(iterations, self.res_u, label='u velocity residual')
            plt.semilogy(iterations, self.res_v, label='v velocity residual')
            plt.semilogy(iterations, self.res_p, label='Pressure correction residual')
            plt.ylabel('Residual')
            plt.xlabel('Iterations')
            plt.grid(True)
            plt.legend()
            plt.savefig(export_path / 'residuals.png')
            plt.close()

        print(f"Results saved to {export_path}")


def main():
    parser = argparse.ArgumentParser(description='Lid-Driven Cavity CFD Solver')
    parser.add_argument('--nx', type=int, default=80, help='Number of cells in x direction')
    parser.add_argument('--ny', type=int, default=80, help='Number of cells in y direction')
    parser.add_argument('--re', type=float, default=100.0, help='Reynolds number')
    parser.add_argument('--iterations', type=int, default=2500, help='Maximum iterations')
    parser.add_argument('--relax_uv', type=float, default=0.8, help='Velocity relaxation factor')
    parser.add_argument('--relax_p', type=float, default=0.1, help='Pressure relaxation factor')
    parser.add_argument('--damping', type=float, default=0.2, help='Inertial damping factor')
    parser.add_argument('--epsilon', type=float, default=1e-10, help='Convergence threshold')

    args = parser.parse_args()

    print(f"Starting Lid-Driven Cavity simulation")
    print(f"Grid: {args.nx} x {args.ny}")
    print(f"Reynolds number: {args.re}")
    print(f"Maximum iterations: {args.iterations}")
    print()

    cavity = LidDrivenCavity(args.nx, args.ny, args.re, args.relax_uv, args.relax_p, args.damping)

    for iteration in range(1, args.iterations + 1):
        print(f"Iteration {iteration} / {args.iterations}")
        cavity.iterate()

        if cavity.has_diverged():
            print("Solution diverged!")
            break

        if cavity.has_converged(args.epsilon):
            print("\nSolution has converged!")
            break

    if not cavity.has_diverged():
        cavity.plot_results(iteration)


if __name__ == "__main__":
    main()