import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, MatrixRankWarning
import warnings


class NSSolver2D:
    """
    2D steady incompressible Navier–Stokes solver using SIMPLE + Rhie–Chow
    Collocated grid, finite-volume, uniform mesh.

    Equations referenced in comments correspond to your report:
      - Convective fluxes: Eq. 10a–10d
      - Diffusive conductance: Eq. 12a–12d
      - Discretized u-momentum: Eq. 13
      - Discretized v-momentum: Eq. 14
      - Upwind conv. coefficients: Eq. 15a–15e
      - Under-relaxation of aP: Eq. 16a, 16b
      - Source terms (pressure gradient): Eq. 17a–17b, 18a–18d
      - Velocity correction coefficients d_u, d_v: Eq. 28, 29
      - Velocity correction formulas: Eq. 30, 31
      - Pressure-correction Poisson eq.: Eq. 33
      - Discretized Poisson coefficients: Eq. 42
      - RHS mass imbalance (continuity error): Eq. 44–45
      - Final correction formulas: Eq. 47a–47c
      - Convergence criteria: Eq. 48a–48c
    """

    def __init__(self, nx, ny, Re, problem_type="cavity",
                 Lx=1.0, Ly=1.0, rho=1.0):
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.problem_type = problem_type
        self.Lx = Lx
        self.Ly = Ly
        self.rho = rho

        # Uniform mesh spacing (non-dimensional) – consistent with Eq. 6a–6f
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)

        # Kinematic viscosity from Re = U_ref * L / nu with U_ref = 1 – Eq. 6f
        self.nu = 1.0 / Re

        # Primary fields (cell-centered)
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))

        # Predictor fields (u*, v*, p')
        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)
        self.p_prime = np.zeros_like(self.p)

        # Momentum aP coefficients for use in d_u, d_v – Eq. 28, 29
        self.aP_u = np.ones_like(self.u)
        self.aP_v = np.ones_like(self.v)

        # Velocity response to pressure, d_u and d_v – Eq. 28, 29
        self.d_u = np.ones_like(self.u) * 1e-10
        self.d_v = np.ones_like(self.v) * 1e-10

        # Under-relaxation factors – Eq. 16a, 16b and in p correction step
        self.alpha_u = 0.7
        self.alpha_v = 0.7
        self.alpha_p = 0.3

        # Solid region mask (for step, back-step) – used in BC description
        self.solid_mask = np.zeros((ny, nx), dtype=bool)
        self._setup_geometry()

    # ------------------------------------------------------------------ #
    # GEOMETRY AND INDEXING
    # ------------------------------------------------------------------ #

    def _setup_geometry(self):
        """
        Set up solid regions depending on problem_type, as per problem setups
        in your report's Figures 1–3.
        """
        if self.problem_type == "cavity":
            # Lid driven cavity – no internal solids
            pass

        elif self.problem_type == "step_cavity":
            # A "step" in the lower-left corner (simple example)
            # Adjust as needed for your exact geometry.
            i_max = int(self.nx / 3)
            j_max = int(self.ny / 3)
            self.solid_mask[:j_max, :i_max] = True

        elif self.problem_type == "backstep":
            # Back-step: lower half blocked in left part of domain
            mid_x = int(self.nx / 2)
            mid_y = int(self.ny / 2)
            self.solid_mask[:mid_y, :mid_x] = True

        else:
            raise ValueError(f"Unknown problem_type: {self.problem_type}")

    def idx(self, i, j):
        """Return flattened index for (i, j) in matrices."""
        return j * self.nx + i

    # ------------------------------------------------------------------ #
    # BOUNDARY CONDITIONS – from BC section at end of Part 1
    # ------------------------------------------------------------------ #

    def apply_velocity_bc(self):
        """
        Apply Dirichlet velocity BCs (no-slip etc.) to u, v.
        Boundaries match the descriptions given for Problems 1–3.
        """
        ny, nx = self.ny, self.nx

        # No-slip on all walls by default (u = 0, v = 0)
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0

        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0

        if self.problem_type in ("cavity", "step_cavity"):
            # Top lid moving with u = 1, v = 0 – consistent with Figure 1
            self.u[-1, :] = 1.0
            self.v[-1, :] = 0.0

        if self.problem_type == "backstep":
            # Inlet: parabolic profile in top half of left boundary
            j_mid = ny // 2
            y = np.linspace(0.0, self.Ly, ny)
            y_top = y[j_mid:]
            h = self.Ly / 2.0
            y_local = y_top - h
            # Simple parabolic profile, max ~1
            u_inlet = 6.0 * (y_local / h) * (1.0 - y_local / h)
            u_inlet = np.clip(u_inlet, 0.0, None)

            self.u[:, 0] = 0.0
            self.u[j_mid:, 0] = u_inlet
            self.v[:, 0] = 0.0

            # Zero-gradient at outlet (approximate) – as described
            self.u[:, -1] = self.u[:, -2]
            self.v[:, -1] = self.v[:, -2]

        # Enforce zero velocity inside solid regions (step, walls)
        self.u[self.solid_mask] = 0.0
        self.v[self.solid_mask] = 0.0

    def apply_pressure_bc(self):
        """
        Apply Neumann-like BCs (zero normal gradient) for pressure on all
        boundaries, with an internal reference cell set to p = 0 – matching
        the description of Neumann BC + reference pressure in the report.
        """
        # Zero-gradient at boundaries (approximate Neumann)
        self.p[0, :] = self.p[1, :]
        self.p[-1, :] = self.p[-2, :]
        self.p[:, 0] = self.p[:, 1]
        self.p[:, -1] = self.p[:, -2]

        # Reference pressure at domain centre (p_ref = 0)
        j_ref = self.ny // 2
        i_ref = self.nx // 2
        self.p[j_ref, i_ref] = 0.0

    # ------------------------------------------------------------------ #
    # MOMENTUM DISCRETIZATION – Eq. 13 and 14
    # ------------------------------------------------------------------ #

    def compute_diffusion_coefficients(self):
        """
        Face diffusive conductances – Eq. 12a–12d.
        Using uniform grid, constant nu.
        """
        dx, dy, nu = self.dx, self.dy, self.nu
        De = nu * dy / dx  # Eq. 12a
        Dw = nu * dy / dx  # Eq. 12b
        Dn = nu * dx / dy  # Eq. 12c
        Ds = nu * dx / dy  # Eq. 12d
        return De, Dw, Dn, Ds

    def assemble_momentum_system(self, component):
        """
        Assemble momentum equation linear system A phi = b for phi = u or v.
        This corresponds to Eq. 13 (for u) and Eq. 14 (for v), using
        upwind convective coefficients (Eq. 15a–15e) and under-relaxation (Eq. 16a–16b).
        """
        nx, ny = self.nx, self.ny
        N = nx * ny
        A = lil_matrix((N, N))
        b = np.zeros(N)

        De, Dw, Dn, Ds = self.compute_diffusion_coefficients()
        dx, dy = self.dx, self.dy
        rho = self.rho

        u = self.u
        v = self.v
        p = self.p

        for j in range(ny):
            for i in range(nx):
                k = self.idx(i, j)

                # Solid region: enforce u = 0 or v = 0 strongly
                if self.solid_mask[j, i]:
                    A[k, k] = 1.0
                    b[k] = 0.0
                    continue

                # Dirichlet velocity BCs at walls / inlet / outlet
                at_boundary = (i == 0 or i == nx - 1 or j == 0 or j == ny - 1)

                if at_boundary:
                    if component == "u":
                        A[k, k] = 1.0
                        b[k] = u[j, i]
                    else:
                        A[k, k] = 1.0
                        b[k] = v[j, i]
                    continue

                # Interior momentum equation
                if component == "u":
                    phi = u
                    # Pressure gradient source term Su – Eq. 17a–17b
                    dpdx = (p[j, i + 1] - p[j, i - 1]) / (2.0 * dx)
                    Su = -dpdx * dy        # Eq. 17a: Su = - (∂p/∂x) Δx Δy / Δx → -dpdx*Δy
                    Sp = 0.0
                else:
                    phi = v
                    # Pressure gradient source term Sv – Eq. 18a–18d
                    dpdy = (p[j + 1, i] - p[j - 1, i]) / (2.0 * dy)
                    Sv = -dpdy * dx        # Eq. 18a: Sv = - (∂p/∂y) Δx Δy / Δy → -dpdy*Δx
                    Sp = 0.0
                    Su = Sv  # reuse name for RHS addition

                # Convective face velocities (simple linear interpolation, no RC here)
                # These are used for upwind convective coefficients – Eq. 15a–15e.
                u_e = 0.5 * (u[j, i] + u[j, i + 1])
                u_w = 0.5 * (u[j, i] + u[j, i - 1])
                v_n = 0.5 * (v[j, i] + v[j + 1, i])
                v_s = 0.5 * (v[j, i] + v[j - 1, i])

                # Convective mass fluxes F = rho * u_f * A – Eq. 10a–10d
                Fe = rho * u_e * dy   # east face
                Fw = rho * u_w * dy   # west face
                Fn = rho * v_n * dx   # north face
                Fs = rho * v_s * dx   # south face

                # Upwind convective + diffusive coefficients – Eq. 15a–15d
                # Sign convention: Fe > 0 is flow P→E, Fw > 0 is W→P, etc.
                aE = De + max(-Fe, 0.0)
                aW = Dw + max(Fw, 0.0)
                aN = Dn + max(-Fn, 0.0)
                aS = Ds + max(Fs, 0.0)

                # Central coefficient – Eq. 15e
                aP = aE + aW + aN + aS - Sp

                # Under-relaxation – Eq. 16a (u), Eq. 16b (v)
                if component == "u":
                    aP_relaxed = aP / self.alpha_u
                    rhs_relax = (1.0 - self.alpha_u) * aP * phi[j, i]
                else:
                    aP_relaxed = aP / self.alpha_v
                    rhs_relax = (1.0 - self.alpha_v) * aP * phi[j, i]

                # Assemble matrix – Eq. 13 or Eq. 14 in discrete form
                A[k, k] = aP_relaxed
                A[k, self.idx(i + 1, j)] = -aE
                A[k, self.idx(i - 1, j)] = -aW
                A[k, self.idx(i, j + 1)] = -aN
                A[k, self.idx(i, j - 1)] = -aS

                b[k] = Su + rhs_relax

        return A.tocsr(), b

    def solve_momentum(self):
        """
        Solve u- and v-momentum equations for predictor fields u* and v*.
        Stores aP_u and aP_v diagonals for d_u, d_v – Eq. 28, 29.
        """
        nx, ny = self.nx, self.ny
        N = nx * ny

        # --- Solve u-momentum (Eq. 13) ---
        A_u, b_u = self.assemble_momentum_system("u")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=MatrixRankWarning)
            u_vec = spsolve(A_u, b_u)
        self.u_star = u_vec.reshape(ny, nx)
        self.aP_u = A_u.diagonal().reshape(ny, nx)

        # --- Solve v-momentum (Eq. 14) ---
        A_v, b_v = self.assemble_momentum_system("v")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=MatrixRankWarning)
            v_vec = spsolve(A_v, b_v)
        self.v_star = v_vec.reshape(ny, nx)
        self.aP_v = A_v.diagonal().reshape(ny, nx)

    # ------------------------------------------------------------------ #
    # SIMPLE d-COEFFICIENTS – Eq. 28, 29
    # ------------------------------------------------------------------ #

    def compute_d_coefficients(self):
        """
        Compute velocity response to pressure, d_u and d_v – Eq. 28, 29.
        In a uniform grid, we can use:
          d_u ~ Δy / aP_u, d_v ~ Δx / aP_v
        """
        eps = 1e-12
        self.d_u = self.dy / (self.aP_u + eps)  # Eq. 28
        self.d_v = self.dx / (self.aP_v + eps)  # Eq. 29

    # ------------------------------------------------------------------ #
    # PRESSURE-CORRECTION (POISSON) – Eq. 33, Eq. 42, Eq. 44–45
    # ------------------------------------------------------------------ #

    def assemble_pressure_correction_system(self):
        """
        Assemble linear system for pressure correction p' based on:
          - Poisson equation for p' – Eq. 33
          - Discretization – Eq. 42
          - RHS mass imbalance – Eq. 44–45
        We also incorporate Rhie–Chow-like face corrections using d_u, d_v
        and the current p and (u*, v*).
        """
        nx, ny = self.nx, self.ny
        N = nx * ny
        A = lil_matrix((N, N))
        b = np.zeros(N)

        dx, dy = self.dx, self.dy
        rho = self.rho

        u_star = self.u_star
        v_star = self.v_star
        d_u = self.d_u
        d_v = self.d_v
        p = self.p

        # Reference cell for p' = 0 (to remove nullspace)
        i_ref = nx // 2
        j_ref = ny // 2

        for j in range(ny):
            for i in range(nx):
                k = self.idx(i, j)

                # Solid cells: p' = 0
                if self.solid_mask[j, i]:
                    A[k, k] = 1.0
                    b[k] = 0.0
                    continue

                # Reference cell: p'_ref = 0
                if i == i_ref and j == j_ref:
                    A[k, k] = 1.0
                    b[k] = 0.0
                    continue

                # --- Compute d-coefficients at faces for Poisson coeffs – Eq. 42 ---

                # East face
                if i < nx - 1 and (not self.solid_mask[j, i + 1]):
                    dE = 0.5 * (d_u[j, i] + d_u[j, i + 1])
                    # aE = dE * Δy / Δx – Eq. 42
                    aE = dE * dy / dx
                else:
                    dE = 0.0
                    aE = 0.0

                # West face
                if i > 0 and (not self.solid_mask[j, i - 1]):
                    dW = 0.5 * (d_u[j, i] + d_u[j, i - 1])
                    # aW = dW * Δy / Δx – Eq. 42
                    aW = dW * dy / dx
                else:
                    dW = 0.0
                    aW = 0.0

                # North face
                if j < ny - 1 and (not self.solid_mask[j + 1, i]):
                    dN = 0.5 * (d_v[j, i] + d_v[j + 1, i])
                    # aN = dN * Δx / Δy – Eq. 42
                    aN = dN * dx / dy
                else:
                    dN = 0.0
                    aN = 0.0

                # South face
                if j > 0 and (not self.solid_mask[j - 1, i]):
                    dS = 0.5 * (d_v[j, i] + d_v[j - 1, i])
                    # aS = dS * Δx / Δy – Eq. 42
                    aS = dS * dx / dy
                else:
                    dS = 0.0
                    aS = 0.0

                # Central coefficient – Eq. 42
                aP = aE + aW + aN + aS

                # Assemble LHS (Poisson stencil) – Eq. 42
                A[k, k] = aP
                if i < nx - 1 and aE != 0.0:
                    A[k, self.idx(i + 1, j)] = -aE
                if i > 0 and aW != 0.0:
                    A[k, self.idx(i - 1, j)] = -aW
                if j < ny - 1 and aN != 0.0:
                    A[k, self.idx(i, j + 1)] = -aN
                if j > 0 and aS != 0.0:
                    A[k, self.idx(i, j - 1)] = -aS

                # --- RHS: mass imbalance from u* and v* – Eq. 44–45 ---

                # Rhie–Chow-like face velocities:
                # u_e' = u*_avg + d_face (p_P - p_E)/Δx – consistent with Eq. 25/30 pattern
                # v_n' = v*_avg + d_face (p_P - p_N)/Δy – consistent with Eq. 27/31

                # East face flux
                if i < nx - 1 and (not self.solid_mask[j, i + 1]):
                    u_e_avg = 0.5 * (u_star[j, i] + u_star[j, i + 1])
                    dE_u = dE
                    u_e = u_e_avg + dE_u * (p[j, i] - p[j, i + 1]) / dx  # Eq. 30 form at face
                    F_e = rho * u_e * dy   # Eq. 10a
                else:
                    F_e = 0.0

                # West face flux
                if i > 0 and (not self.solid_mask[j, i - 1]):
                    u_w_avg = 0.5 * (u_star[j, i] + u_star[j, i - 1])
                    dW_u = dW
                    u_w = u_w_avg + dW_u * (p[j, i] - p[j, i - 1]) / dx  # Eq. 30 form at face
                    F_w = rho * u_w * dy   # Eq. 10b
                else:
                    F_w = 0.0

                # North face flux
                if j < ny - 1 and (not self.solid_mask[j + 1, i]):
                    v_n_avg = 0.5 * (v_star[j, i] + v_star[j + 1, i])
                    dN_v = dN
                    v_n = v_n_avg + dN_v * (p[j, i] - p[j + 1, i]) / dy  # Eq. 31 form at face
                    F_n = rho * v_n * dx   # Eq. 10c
                else:
                    F_n = 0.0

                # South face flux
                if j > 0 and (not self.solid_mask[j - 1, i]):
                    v_s_avg = 0.5 * (v_star[j, i] + v_star[j - 1, i])
                    dS_v = dS
                    v_s = v_s_avg + dS_v * (p[j, i] - p[j - 1, i]) / dy  # Eq. 31 form at face
                    F_s = rho * v_s * dx   # Eq. 10d
                else:
                    F_s = 0.0

                # Mass imbalance at cell P – Eq. 44–45:
                # b_P = (F_e - F_w) + (F_n - F_s)
                mass_imbalance = (F_e - F_w) + (F_n - F_s)
                b[k] = mass_imbalance

        return A.tocsr(), b

    def solve_pressure_correction(self):
        """
        Solve Poisson equation for p' – Eq. 33, discretized as Eq. 42,
        with source term from mass imbalance – Eq. 44–45.
        Then update p using Eq. 47c.
        """
        nx, ny = self.nx, self.ny

        A_p, b_p = self.assemble_pressure_correction_system()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=MatrixRankWarning)
            p_vec = spsolve(A_p, b_p)
        self.p_prime = p_vec.reshape(ny, nx)

        # Correct pressure – Eq. 47c: p_new = p_old + α_p * p'
        self.p += self.alpha_p * self.p_prime

    # ------------------------------------------------------------------ #
    # VELOCITY CORRECTION – Eq. 30, 31, 47a, 47b
    # ------------------------------------------------------------------ #

    def correct_velocities(self):
        """
        Apply velocity corrections:
          u = u* + d_u (p'_W - p'_E)/Δx – Eq. 30, 47a
          v = v* + d_v (p'_S - p'_N)/Δy – Eq. 31, 47b
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        self.u[:, :] = self.u_star
        self.v[:, :] = self.v_star

        for j in range(ny):
            for i in range(nx):
                if self.solid_mask[j, i]:
                    self.u[j, i] = 0.0
                    self.v[j, i] = 0.0
                    continue

                # Neighbour pressure corrections
                pE = self.p_prime[j, i + 1] if i + 1 < nx else self.p_prime[j, i]
                pW = self.p_prime[j, i - 1] if i - 1 >= 0 else self.p_prime[j, i]
                pN = self.p_prime[j + 1, i] if j + 1 < ny else self.p_prime[j, i]
                pS = self.p_prime[j - 1, i] if j - 1 >= 0 else self.p_prime[j, i]

                # Velocity corrections – Eq. 30, 31
                self.u[j, i] += self.d_u[j, i] * (pW - pE) / dx  # Eq. 30 / 47a
                self.v[j, i] += self.d_v[j, i] * (pS - pN) / dy  # Eq. 31 / 47b

    # ------------------------------------------------------------------ #
    # DIVERGENCE AND CONVERGENCE CHECK – Eq. 48a–48c
    # ------------------------------------------------------------------ #

    def compute_divergence(self):
        """
        Compute discrete divergence at cell centres from u, v.
        Used for Eq. 48c: max |∇·u| < tolerance.
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        div = np.zeros_like(self.u)

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                du_dx = (self.u[j, i + 1] - self.u[j, i - 1]) / (2.0 * dx)
                dv_dy = (self.v[j + 1, i] - self.v[j - 1, i]) / (2.0 * dy)
                div[j, i] = du_dx + dv_dy

        return div

    # ------------------------------------------------------------------ #
    # SIMPLE LOOP – Putting it all together
    # ------------------------------------------------------------------ #

    def solve(self, max_iter=2000, check_every=5,
              tol_vel=1e-6, tol_div=1e-4, debug=True):
        """
        SIMPLE algorithm loop:
          1. Solve momentum (u*, v*) – Eq. 19–20 in your narrative
          2. Compute d_u, d_v – Eq. 28–29
          3. Solve pressure-correction Poisson – Eq. 33 / 42
          4. Correct u, v, p – Eq. 47a–47c
          5. Check convergence – Eq. 48a–48c
        """
        self.apply_velocity_bc()
        self.apply_pressure_bc()

        for it in range(max_iter):
            u_old = self.u.copy()
            v_old = self.v.copy()

            # 1) Momentum predictor step – Eqs. 13, 14, 19–20
            self.solve_momentum()

            # 2) Compute d-coefficients – Eq. 28, 29
            self.compute_d_coefficients()

            # 3) Pressure correction – Eqs. 33, 42, 44–45
            self.solve_pressure_correction()

            # 4) Velocity correction – Eq. 30, 31, 47a, 47b
            self.correct_velocities()

            # 5) Re-apply BCs (to keep walls/inlet/outlet consistent)
            self.apply_velocity_bc()
            self.apply_pressure_bc()

            # 6) Convergence checks – Eq. 48a–48c
            if it % check_every == 0 or it == 0:
                diff_u = np.max(np.abs(self.u - u_old))  # Eq. 48a
                diff_v = np.max(np.abs(self.v - v_old))  # Eq. 48b
                max_diff = max(diff_u, diff_v)
                div = self.compute_divergence()
                max_div = np.max(np.abs(div))            # Eq. 48c

                if debug:
                    print(
                        f"Iter {it:5d}: "
                        f"max_diff = {max_diff: .3e}, "
                        f"max_div = {max_div: .3e}, "
                        f"p_prime_max = {np.max(np.abs(self.p_prime)): .3e}"
                    )

                if max_diff < tol_vel and max_div < tol_div:
                    if debug:
                        print(
                            f"CONVERGED at iteration {it} "
                            f"(max_diff={max_diff:.3e}, max_div={max_div:.3e})"
                        )
                    break
        else:
            if debug:
                print("WARNING: SIMPLE did NOT converge within max_iter.")

    # ------------------------------------------------------------------ #
    # SIMPLE PLOTTING FUNCTIONS
    # ------------------------------------------------------------------ #

    def plot_velocity_magnitude(self):
        speed = np.sqrt(self.u**2 + self.v**2)
        X, Y = np.meshgrid(
            np.linspace(0, self.Lx, self.nx),
            np.linspace(0, self.Ly, self.ny)
        )
        plt.figure()
        plt.contourf(X, Y, speed, levels=40)
        plt.colorbar(label="|u|")
        plt.title(f"Velocity magnitude ({self.problem_type})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    def plot_streamlines(self, density=1.5):
        X, Y = np.meshgrid(
            np.linspace(0, self.Lx, self.nx),
            np.linspace(0, self.Ly, self.ny)
        )
        plt.figure()
        plt.streamplot(X, Y, self.u, self.v, density=density)
        plt.title(f"Streamlines ({self.problem_type})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------- #
# DRIVER / EXAMPLE
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    # Pick one of: "cavity", "step_cavity", "backstep"
    problem = "cavity"
    nx, ny = 65, 65
    Re = 100.0

    solver = NSSolver2D(nx=nx, ny=ny, Re=Re, problem_type=problem)

    solver.solve(max_iter=2000,
                 check_every=5,
                 tol_vel=1e-6,
                 tol_div=1e-4,
                 debug=True)

    solver.plot_velocity_magnitude()
    solver.plot_streamlines()
