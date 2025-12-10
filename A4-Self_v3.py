import numpy as np
import matplotlib.pyplot as plt
import math

# Works for smaller grids (30x30).

class solver:
    def __init__(self, nx, ny, Re, problem_type):
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.problem_type = problem_type
        self.nu = 1.0 / self.Re

        # Velocity and pressure fields
        # +2 as need ghost cells. ny, nx for the grid size, one ghost cell on each side
        self.u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        self.v = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        self.p = np.zeros((ny + 2, nx + 2), dtype=np.float64)

        # Correction fields
        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)
        self.p_star = np.zeros_like(self.p)
        self.p_prime = np.zeros_like(self.p)

        # Momentum Coefficients
        self.a_p = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.a_e = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.a_w = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.a_n = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.a_s = np.ones((ny + 2, nx + 2), dtype=np.float64)

        # Pressure Correction Coefficients
        self.Ap_e = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.Ap_w = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.Ap_n = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.Ap_s = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.Ap_p = np.ones((ny + 2, nx + 2), dtype=np.float64)

        # Source Terms
        self.b_u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        self.b_v = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        self.b_p = np.zeros((ny + 2, nx + 2), dtype=np.float64)

        # Face Velocities
        self.u_face = np.zeros((ny + 2, nx + 1), dtype=np.float64)
        self.v_face = np.zeros((ny + 1, nx + 2), dtype=np.float64)

        # Under relaxation variables. Was 0.7 and 0.3 for a 30x30 grid.
        self.alpha_uv = 0.25
        self.alpha_p = 0.1

        # Convergence criteria for iterative solvers
        self.tolerance_uv = 1e-3
        self.tolerance_p = 1e-4
        # was 50, 200, 250. Multiply by 8.5 for the larger grid
        self.max_iterations_uv = 300
        self.max_iterations_p = 1000
        self.max_outer_iterations = 2125

        self._setup_geometry()

        # Boundary Conditions
        self.u[0, 1:self.nx + 1] = 1.0
        self.u_star[0, 1:self.nx + 1] = 1.0
        self.u_face[0, 1:self.nx] = 1.0

    def _setup_geometry(self):
        if self.problem_type == 'cavity':
            self.Lx = 1.0
            self.Ly = 1.0
            self.dx = self.Lx / self.nx
            self.dy = self.Ly / self.ny

            # Mesh Creation
            x = np.array([0.0], dtype=np.float64)
            y = np.array([0.0], dtype=np.float64)

            x = np.append(x, np.linspace(self.dx / 2, 1 - self.dx / 2, self.nx))
            x = np.append(x, [1.0])

            y = np.append(y, np.linspace(self.dy / 2, 1 - self.dy / 2, self.ny))
            y = np.append(y, [1.0])

            self.x, self.y = np.meshgrid(x, y)

    def _compute_diffusion_coefficient(self):
        # Equations 12a - 12d
        De = self.dy / (self.Re * self.dx)
        Dw = self.dy / (self.Re * self.dx)
        Dn = self.dx / (self.Re * self.dy)
        Ds = self.dx / (self.Re * self.dy)

        return De, Dw, Dn, Ds

    def _compute_convective_mass(self, i, j):
        # Convective mass flux coefficients, Equation 10a - 10d
        Fe = self.dy * self.u_face[i, j]
        Fw = self.dy * self.u_face[i, j - 1]
        Fn = self.dx * self.v_face[i - 1, j]
        Fs = self.dx * self.v_face[i, j]

        return Fe, Fw, Fn, Fs

    def _solve_momentum(self):
        De, Dw, Dn, Ds = self._compute_diffusion_coefficient()

        # Interior cells
        for i in range(2, self.ny):
            for j in range(2, self.nx):
                Fe, Fw, Fn, Fs = self._compute_convective_mass(i, j)
                self.a_e[i, j] = De + max(0.0, -Fe)
                self.a_w[i, j] = Dw + max(0.0, Fw)
                self.a_n[i, j] = Dn + max(0.0, -Fn)
                self.a_s[i, j] = Ds + max(0.0, Fs)
                self.a_p[i, j] = self.a_e[i, j] + self.a_w[i, j] + self.a_n[i, j] + self.a_s[i, j] + (Fe - Fw) + (
                            Fn - Fs)
                self.b_u[i, j] = 0.5 * (self.p[i, j - 1] - self.p[i, j + 1]) * self.dx
                self.b_v[i, j] = 0.5 * (self.p[i + 1, j] - self.p[i - 1, j]) * self.dy

        # Left Wall
        j = 1
        for i in range(2, self.ny):
            Fe, Fw, Fn, Fs = self._compute_convective_mass(i, j)
            self.a_e[i, j] = De + max(0.0, -Fe)
            self.a_w[i, j] = 2 * Dw + max(0.0, Fw)
            self.a_n[i, j] = Dn + max(0.0, -Fn)
            self.a_s[i, j] = Ds + max(0.0, Fs)
            self.a_p[i, j] = self.a_e[i, j] + self.a_w[i, j] + self.a_n[i, j] + self.a_s[i, j] + (Fe - Fw) + (Fn - Fs)
            self.b_u[i, j] = 0.5 * (self.p[i, j] - self.p[i, j + 1]) * self.dx
            self.b_v[i, j] = 0.5 * (self.p[i + 1, j] - self.p[i - 1, j]) * self.dy

        # Bottom Wall
        i = self.ny
        for j in range(2, self.nx):
            Fe, Fw, Fn, Fs = self._compute_convective_mass(i, j)
            self.a_e[i, j] = De + max(0.0, -Fe)
            self.a_w[i, j] = Dw + max(0.0, Fw)
            self.a_n[i, j] = Dn + max(0.0, -Fn)
            self.a_s[i, j] = 2 * Ds + max(0.0, Fs)
            self.a_p[i, j] = self.a_e[i, j] + self.a_w[i, j] + self.a_n[i, j] + self.a_s[i, j] + (Fe - Fw) + (Fn - Fs)
            self.b_u[i, j] = 0.5 * (self.p[i, j - 1] - self.p[i, j + 1]) * self.dx
            self.b_v[i, j] = 0.5 * (self.p[i, j] - self.p[i - 1, j]) * self.dy

        # Right Wall
        j = self.nx
        for i in range(2, self.ny):
            Fe, Fw, Fn, Fs = self._compute_convective_mass(i, j)
            self.a_e[i, j] = De + max(0.0, -Fe)
            self.a_w[i, j] = 2 * Dw + max(0.0, Fw)
            self.a_n[i, j] = Dn + max(0.0, -Fn)
            self.a_s[i, j] = Ds + max(0.0, Fs)
            self.a_p[i, j] = self.a_e[i, j] + self.a_w[i, j] + self.a_n[i, j] + self.a_s[i, j] + (Fe - Fw) + (Fn - Fs)
            self.b_u[i, j] = 0.5 * (self.p[i, j - 1] - self.p[i, j]) * self.dx
            self.b_v[i, j] = 0.5 * (self.p[i + 1, j] - self.p[i - 1, j]) * self.dy

        # Top Wall
        i = 1
        for j in range(2, self.ny):
            Fe, Fw, Fn, Fs = self._compute_convective_mass(i, j)
            self.a_e[i, j] = De + max(0.0, -Fe)
            self.a_w[i, j] = Dw + max(0.0, Fw)
            self.a_n[i, j] = 2 * Dn + max(0.0, -Fn)
            self.a_s[i, j] = Ds + max(0.0, Fs)
            self.a_p[i, j] = self.a_e[i, j] + self.a_w[i, j] + self.a_n[i, j] + self.a_s[i, j] + (Fe - Fw) + (Fn - Fs)
            self.b_u[i, j] = 0.5 * (self.p[i, j - 1] - self.p[i, j + 1]) * self.dx
            self.b_v[i, j] = 0.5 * (self.p[i + 1, j] - self.p[i, j]) * self.dy

        # Top Left Corner
        i = 1
        j = 1
        Fe, Fw, Fn, Fs = self._compute_convective_mass(i, j)
        self.a_e[i, j] = De + max(0.0, -Fe)
        self.a_w[i, j] = 2 * Dw + max(0.0, Fw)
        self.a_n[i, j] = 2 * Dn + max(0.0, -Fn)
        self.a_s[i, j] = Ds + max(0.0, Fs)
        self.a_p[i, j] = self.a_e[i, j] + self.a_w[i, j] + self.a_n[i, j] + self.a_s[i, j] + (Fe - Fw) + (Fn - Fs)
        self.b_u[i, j] = 0.5 * (self.p[i, j] - self.p[i, j + 1]) * self.dx
        self.b_v[i, j] = 0.5 * (self.p[i + 1, j] - self.p[i, j]) * self.dy

        # Top Right Corner
        i = 1
        j = self.nx
        Fe, Fw, Fn, Fs = self._compute_convective_mass(i, j)
        self.a_e[i, j] = De + max(0.0, -Fe)
        self.a_w[i, j] = 2 * Dw + max(0.0, Fw)
        self.a_n[i, j] = 2 * Dn + max(0.0, -Fn)
        self.a_s[i, j] = Ds + max(0.0, Fs)
        self.a_p[i, j] = self.a_e[i, j] + self.a_w[i, j] + self.a_n[i, j] + self.a_s[i, j] + (Fe - Fw) + (Fn - Fs)
        self.b_u[i, j] = 0.5 * (self.p[i, j - 1] - self.p[i, j]) * self.dx
        self.b_v[i, j] = 0.5 * (self.p[i + 1, j] - self.p[i, j]) * self.dy

        # Bottom Left Corner
        i = self.ny
        j = 1
        Fe, Fw, Fn, Fs = self._compute_convective_mass(i, j)
        self.a_e[i, j] = De + max(0.0, -Fe)
        self.a_w[i, j] = 2 * Dw + max(0.0, Fw)
        self.a_n[i, j] = Dn + max(0.0, -Fn)
        self.a_s[i, j] = 2 * Ds + max(0.0, Fs)
        self.a_p[i, j] = self.a_e[i, j] + self.a_w[i, j] + self.a_n[i, j] + self.a_s[i, j] + (Fe - Fw) + (Fn - Fs)
        self.b_u[i, j] = 0.5 * (self.p[i, j] - self.p[i, j + 1]) * self.dx
        self.b_v[i, j] = 0.5 * (self.p[i, j] - self.p[i - 1, j]) * self.dy

        # Bottom Right Corner
        i = self.ny
        j = self.nx
        Fe, Fw, Fn, Fs = self._compute_convective_mass(i, j)
        self.a_e[i, j] = 2 * De + max(0.0, -Fe)
        self.a_w[i, j] = 2 * Dw + max(0.0, Fw)
        self.a_n[i, j] = Dn + max(0.0, -Fn)
        self.a_s[i, j] = Ds + max(0.0, Fs)
        self.a_p[i, j] = self.a_e[i, j] + self.a_w[i, j] + self.a_n[i, j] + self.a_s[i, j] + (Fe - Fw) + (Fn - Fs)
        self.b_u[i, j] = 0.5 * (self.p[i, j - 1] - self.p[i, j]) * self.dx
        self.b_v[i, j] = 0.5 * (self.p[i, j] - self.p[i - 1, j]) * self.dy

    def _solve_uv_matrix(self):
        # Solve u momentum
        for n_u in range(1, self.max_iterations_uv + 1):
            error_u = 0
            for i in range(1, self.ny + 1):
                for j in range(1, self.nx + 1):
                    u_old = self.u[i,j]
                    self.u[i, j] = self.alpha_uv * (
                            self.a_e[i, j] * self.u[i, j + 1] +
                            self.a_w[i, j] * self.u[i, j - 1] +
                            self.a_n[i, j] * self.u[i - 1, j] +
                            self.a_s[i, j] * self.u[i + 1, j] +
                            self.b_u[i, j]) / self.a_p[i, j] + (1 - self.alpha_uv) * self.u_star[i, j]
                    error_u += (self.u[i,j] - u_old) ** 2
            if n_u == 1:
                norm_u = math.sqrt(error_u)
            error_u = math.sqrt(error_u)

            if error_u < self.tolerance_uv and norm_u > 1e-10:
                break

        # Solve v momentum
        for n_v in range(1, self.max_iterations_uv + 1):
            error_v = 0
            for i in range(1, self.ny + 1):
                for j in range(1, self.nx + 1):
                    v_old = self.v[i,j]
                    self.v[i, j] = self.alpha_uv * (
                            self.a_e[i, j] * self.v[i, j + 1] +
                            self.a_w[i, j] * self.v[i, j - 1] +
                            self.a_n[i, j] * self.v[i - 1, j] +
                            self.a_s[i, j] * self.v[i + 1, j] +
                            self.b_v[i, j]) / self.a_p[i, j] + (1 - self.alpha_uv) * self.v_star[i, j]
                    error_v += (self.v[i,j] - v_old) ** 2

            if n_v == 1:
                norm_v = math.sqrt(error_v)
            error_v = math.sqrt(error_v)

            if error_v < self.tolerance_uv and norm_v > 1e-10:
                break

        return error_u, error_v

    def _face_velocity(self):
        # U face velocity
        for i in range(1, self.ny + 1):
            for j in range(1, self.nx):
                self.u_face[i, j] = 0.5 * (self.u[i, j] + self.u[i, j + 1]) + \
                                    0.25 * self.alpha_uv * (self.p[i, j + 1] - self.p[i, j - 1]) * self.dy / self.a_p[
                                        i, j] + \
                                    0.25 * self.alpha_uv * (self.p[i, j + 2] - self.p[i, j]) * self.dy / self.a_p[
                                        i, j + 1] - \
                                    0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j + 1]) * (
                                                self.p[i, j + 1] - self.p[i, j]) * self.dy

        # V face velocity
        for i in range(2, self.ny + 1):
            for j in range(1, self.nx + 1):
                self.v_face[i - 1, j] = 0.5 * (self.v[i, j] + self.v[i - 1, j]) + \
                                        0.25 * self.alpha_uv * (self.p[i - 1, j] - self.p[i + 1, j]) * self.dy / \
                                        self.a_p[i, j] + \
                                        0.25 * self.alpha_uv * (self.p[i - 2, j] - self.p[i, j]) * self.dy / self.a_p[
                                            i - 1, j] - \
                                        0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i - 1, j]) * (
                                                    self.p[i - 1, j] - self.p[i, j]) * self.dy

    def _pressure_correction(self):
        # Interior cells
        for i in range(2, self.ny):
            for j in range(2, self.nx):
                self.Ap_e[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j + 1]) * (self.dy ** 2)
                self.Ap_w[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j - 1]) * (self.dy ** 2)
                self.Ap_n[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i - 1, j]) * (self.dx ** 2)
                self.Ap_s[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i + 1, j]) * (self.dx ** 2)
                self.Ap_p[i, j] = self.Ap_e[i, j] + self.Ap_w[i, j] + self.Ap_n[i, j] + self.Ap_s[i, j]
                self.b_p[i, j] = -(self.u_face[i, j] - self.u_face[i, j - 1]) * self.dy - (
                            self.v_face[i - 1, j] - self.v_face[i, j]) * self.dx

        # Top
        i = 1
        for j in range(2, self.nx):
            self.Ap_e[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j + 1]) * (self.dy ** 2)
            self.Ap_w[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j - 1]) * (self.dy ** 2)
            self.Ap_n[i, j] = 0
            self.Ap_s[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i + 1, j]) * (self.dx ** 2)
            self.Ap_p[i, j] = self.Ap_e[i, j] + self.Ap_w[i, j] + self.Ap_n[i, j] + self.Ap_s[i, j]
            self.b_p[i, j] = -(self.u_face[i, j] - self.u_face[i, j - 1]) * self.dy - (
                        self.v_face[i - 1, j] - self.v_face[i, j]) * self.dx

        # Left
        j = 1
        for i in range(2, self.ny):
            self.Ap_e[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j + 1]) * (self.dy ** 2)
            self.Ap_w[i, j] = 0
            self.Ap_n[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i - 1, j]) * (self.dx ** 2)
            self.Ap_s[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i + 1, j]) * (self.dx ** 2)
            self.Ap_p[i, j] = self.Ap_e[i, j] + self.Ap_w[i, j] + self.Ap_n[i, j] + self.Ap_s[i, j]
            self.b_p[i, j] = -(self.u_face[i, j] - self.u_face[i, j - 1]) * self.dy - (
                        self.v_face[i - 1, j] - self.v_face[i, j]) * self.dx

        # Right
        j = self.nx
        for i in range(2, self.ny):
            self.Ap_e[i, j] = 0
            self.Ap_w[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j - 1]) * (self.dy ** 2)
            self.Ap_n[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i - 1, j]) * (self.dx ** 2)
            self.Ap_s[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i + 1, j]) * (self.dx ** 2)
            self.Ap_p[i, j] = self.Ap_e[i, j] + self.Ap_w[i, j] + self.Ap_n[i, j] + self.Ap_s[i, j]
            self.b_p[i, j] = -(self.u_face[i, j] - self.u_face[i, j - 1]) * self.dy - (
                        self.v_face[i - 1, j] - self.v_face[i, j]) * self.dx

        # Bottom
        i = self.ny
        for j in range(2, self.nx):
            self.Ap_e[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j + 1]) * (self.dy ** 2)
            self.Ap_w[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j - 1]) * (self.dy ** 2)
            self.Ap_n[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i - 1, j]) * (self.dx ** 2)
            self.Ap_s[i, j] = 0
            self.Ap_p[i, j] = self.Ap_e[i, j] + self.Ap_w[i, j] + self.Ap_n[i, j] + self.Ap_s[i, j]
            self.b_p[i, j] = -(self.u_face[i, j] - self.u_face[i, j - 1]) * self.dy - (
                        self.v_face[i - 1, j] - self.v_face[i, j]) * self.dx

        # Top Left Corner
        i = 1
        j = 1
        self.Ap_e[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j + 1]) * (self.dy ** 2)
        self.Ap_w[i, j] = 0
        self.Ap_n[i, j] = 0
        self.Ap_s[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i + 1, j]) * (self.dx ** 2)
        self.Ap_p[i, j] = self.Ap_e[i, j] + self.Ap_w[i, j] + self.Ap_n[i, j] + self.Ap_s[i, j]
        self.b_p[i, j] = -(self.u_face[i, j] - self.u_face[i, j - 1]) * self.dy - (
                    self.v_face[i - 1, j] - self.v_face[i, j]) * self.dx

        # Top Right Corner
        i = 1
        j = self.nx
        self.Ap_e[i, j] = 0
        self.Ap_w[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j - 1]) * (self.dy ** 2)
        self.Ap_n[i, j] = 0
        self.Ap_s[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i + 1, j]) * (self.dx ** 2)
        self.Ap_p[i, j] = self.Ap_e[i, j] + self.Ap_w[i, j] + self.Ap_n[i, j] + self.Ap_s[i, j]
        self.b_p[i, j] = -(self.u_face[i, j] - self.u_face[i, j - 1]) * self.dy - (
                    self.v_face[i - 1, j] - self.v_face[i, j]) * self.dx

        # Bottom Left Corner
        i = self.ny
        j = 1
        self.Ap_e[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j + 1]) * (self.dy ** 2)
        self.Ap_w[i, j] = 0
        self.Ap_n[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i - 1, j]) * (self.dx ** 2)
        self.Ap_s[i, j] = 0
        self.Ap_p[i, j] = self.Ap_e[i, j] + self.Ap_w[i, j] + self.Ap_n[i, j] + self.Ap_s[i, j]
        self.b_p[i, j] = -(self.u_face[i, j] - self.u_face[i, j - 1]) * self.dy - (
                    self.v_face[i - 1, j] - self.v_face[i, j]) * self.dx

        # Bottom Right Corner
        i = self.ny
        j = self.nx
        self.Ap_e[i, j] = 0
        self.Ap_w[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i, j - 1]) * (self.dy ** 2)
        self.Ap_n[i, j] = 0.5 * self.alpha_uv * (1 / self.a_p[i, j] + 1 / self.a_p[i - 1, j]) * (self.dx ** 2)
        self.Ap_s[i, j] = 0
        self.Ap_p[i, j] = self.Ap_e[i, j] + self.Ap_w[i, j] + self.Ap_n[i, j] + self.Ap_s[i, j]
        self.b_p[i, j] = -(self.u_face[i, j] - self.u_face[i, j - 1]) * self.dy - (
                    self.v_face[i - 1, j] - self.v_face[i, j]) * self.dx

    def _solve_p_matrix(self):
        for n_p in range(1, self.max_iterations_p + 1):
            error_p = 0
            for i in range(1, self.ny + 1):
                for j in range(1, self.nx + 1):
                    p_old = self.p_prime[i,j]
                    self.p_prime[i, j] = (self.Ap_e[i, j] * self.p_prime[i, j + 1] +
                                                 self.Ap_w[i, j] * self.p_prime[i, j - 1] +
                                                 self.Ap_n[i, j] * self.p_prime[i - 1, j] +
                                                 self.Ap_s[i, j] * self.p_prime[i + 1, j] +
                                                 self.b_p[i, j]) / self.Ap_p[i, j]
                    error_p += (self.p_prime[i,j] - p_old) ** 2

            if n_p == 1:
                norm_p = math.sqrt(error_p)
            error_p = math.sqrt(error_p)

            if norm_p > 1e-10 and error_p < self.tolerance_p:
                break

        return error_p

    def _correct_pressure(self):
        self.p_star = self.p + self.alpha_p * self.p_prime

        # BC - Top wall
        self.p_star[0, 1:self.nx + 1] = self.p_star[1, 1:self.nx + 1]
        # Left wall
        self.p_star[1:self.ny + 1, 0] = self.p_star[1:self.ny + 1, 1]
        # Right wall
        self.p_star[1:self.ny + 1, self.nx + 1] = self.p_star[1:self.ny + 1, self.nx]
        # Bottom wall
        self.p_star[self.ny + 1, 1:self.nx + 1] = self.p_star[self.ny, 1:self.nx + 1]

        # Top left corner
        self.p_star[0, 0] = (self.p_star[1, 2] + self.p_star[0, 1] + self.p_star[1, 0]) / 3
        # Top right corner
        self.p_star[0, self.nx + 1] = (self.p_star[0, self.nx] + self.p_star[1, self.nx] + self.p_star[
            1, self.nx + 1]) / 3
        # Bottom left corner
        self.p_star[self.ny + 1, 0] = (self.p_star[self.ny, 0] + self.p_star[self.ny, 1] + self.p_star[
            self.ny + 1, 1]) / 3
        # Bottom right corner
        self.p_star[self.ny + 1, self.nx + 1] = (self.p_star[self.ny, self.nx + 1] + self.p_star[self.ny + 1, self.nx] +
                                                 self.p_star[self.ny, self.nx]) / 3

    def _correct_cell_center_vel(self):
        # u velocity - interior cells
        for i in range(1, self.ny + 1):
            for j in range(2, self.nx):
                self.u_star[i, j] = self.u[i, j] + 0.5 * self.alpha_uv * (
                            self.p_prime[i, j - 1] - self.p_prime[i, j + 1]) * self.dy / self.a_p[i, j]

        # u velocity - left boundary
        j = 1
        for i in range(1, self.ny + 1):
            self.u_star[i, j] = self.u[i, j] + 0.5 * self.alpha_uv * (
                        self.p_prime[i, j] - self.p_prime[i, j + 1]) * self.dy / self.a_p[i, j]

        # u velocity - right boundary
        j = self.nx
        for i in range(1, self.ny + 1):
            self.u_star[i, j] = self.u[i, j] + 0.5 * self.alpha_uv * (
                        self.p_prime[i, j - 1] - self.p_prime[i, j]) * self.dy / self.a_p[i, j]

        # v velocity - interior cells
        for i in range(2, self.ny):
            for j in range(1, self.nx + 1):
                self.v_star[i, j] = self.v[i, j] + 0.5 * self.alpha_uv * (
                            self.p_prime[i + 1, j] - self.p_prime[i - 1, j]) * self.dx / self.a_p[i, j]

        # v velocity - top boundary
        i = 1
        for j in range(1, self.nx + 1):
            self.v_star[i, j] = self.v[i, j] + 0.5 * self.alpha_uv * (
                        self.p_prime[i + 1, j] - self.p_prime[i, j]) * self.dx / self.a_p[i, j]

        # v velocity - bottom boundary
        i = self.ny
        for j in range(1, self.nx + 1):
            self.v_star[i, j] = self.v[i, j] + 0.5 * self.alpha_uv * (
                        self.p_prime[i, j] - self.p_prime[i - 1, j]) * self.dx / self.a_p[i, j]

    def _correct_face_velocity(self):
        for i in range(1, self.ny + 1):
            for j in range(1, self.nx):
                self.u_face[i, j] = self.u_face[i, j] + 0.5 * self.alpha_uv * (
                            1 / self.a_p[i, j] + 1 / self.a_p[i, j + 1]) * (
                                                self.p_prime[i, j] - self.p_prime[i, j + 1]) * self.dy

        for i in range(2, self.ny + 1):
            for j in range(1, self.nx + 1):
                self.v_face[i - 1, j] = self.v_face[i - 1, j] + 0.5 * self.alpha_uv * (
                            1 / self.a_p[i, j] + 1 / self.a_p[i - 1, j]) * (
                                                    self.p_prime[i, j] - self.p_prime[i - 1, j]) * self.dx

    def SIMPLE(self):
        for n_simple in range(1, self.max_outer_iterations + 1):
            self._solve_momentum()
            error_u, error_v = self._solve_uv_matrix()
            self._face_velocity()
            self._pressure_correction()
            error_p = self._solve_p_matrix()

            mass_res = np.max(np.abs(self.b_p[1:-1, 1:-1]))

            self._correct_pressure()
            self._correct_cell_center_vel()
            self._correct_face_velocity()
            self.p = np.copy(self.p_star)

            if n_simple % 10 == 0:  # Print every 10 iterations
                print(f'Iteration {n_simple}: u: {error_u}, v:{error_v}, p:{error_p}, mass: {mass_res}')

            if (error_u < self.tolerance_uv and error_v < self.tolerance_uv and error_p < self.tolerance_p and mass_res < 1e-6):
                print(f"Converged on iteration {n_simple}!")
                print(f'Iteration {n_simple}: u: {error_u}, v:{error_v}, p:{error_p}, mass: {mass_res}')

                break

        # Plot streamlines
        plt.figure(figsize=(8, 8))
        x_plot = self.x[1:-1, 1:-1]
        y_plot = self.y[1:-1, 1:-1]
        u_plot = np.flipud(self.u_star[1:-1, 1:-1])
        v_plot = np.flipud(self.v_star[1:-1, 1:-1])
        plt.streamplot(x_plot, y_plot, u_plot, v_plot, density=1.8, linewidth=1, arrowsize=1)
        plt.title('Streamlines')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


if __name__ == "__main__":
    cavitySolve = solver(nx=257, ny=257, Re=100, problem_type='cavity')
    cavitySolve.SIMPLE()