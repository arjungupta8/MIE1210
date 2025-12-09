import numpy as np
import matplotlib.pyplot as plt
import math

from A4_Test import max_inner_iteration_uv


class solver:
    def __init__(self, nx, ny, Re, problem_type):
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.problem_type = problem_type
        self.nu = 1.0/ self.Re

        # Velocity and pressure fields
        # +2 as need ghost cells. ny, nx for the grid size, one ghost cell on each side
        self.u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        self.v = np.zeros((ny + 2, nx + 2), dtype = np.float64)
        self.p = np.zeros((ny + 2, nx + 2), dtype = np.float64)

        # Correction fields
        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)
        self.p_star = np.zeros_like(self.p)
        self.p_prime = np.zeros_like(self.p)

        # Convective Coefficients
        self.a_p = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.a_e = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.a_w = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.a_n = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        self.a_s = np.zeros((ny + 2, nx + 2), dtype=np.float64)

        # Pressure Correction Coefficients
        self.Ap_e = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.Ap_w = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.Ap_n = np.ones((ny + 2, nx + 2), dtype=np.float64)
        self.Ap_s = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        self.Ap_p = np.ones((ny + 2, nx + 2), dtype=np.float64)

        # Source Terms
        self.b_u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        self.b_v = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        self.b_p = np.zeros((ny + 2, nx + 2), dtype=np.float64)

        # Face Velocities
        self.u_face = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        self.v_face = np.zeros((ny + 2, nx + 2), dtype=np.float64)

        # D coefficients
        self.d_u = np.ones((ny, nx)) * 1e-10
        self.d_v = np.ones((ny, nx)) * 1e-10

        # Under relaxation variables
        self.alpha_uv = 0.7
        self.alpha_p = 0.3

        # Convergence criteria for iterative solvers
        self.tolerance = 1e-4
        self.max_iterations = 250
        self.check_interval = 5



        self.solid_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self._setup_geometry()

        self._apply_bc()

    def _setup_geometry(self):
        if self.problem_type == 'cavity':
            self.Lx = 1.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx)
            self.dy = self.Ly / (self.ny)

            # Mesh Creation
            # Generate internal cell centers
            x_internal = np.linspace(self.dx / 2, 1 - self.dx / 2, self.nx)
            y_internal = np.linspace(self.dy / 2, 1 - self.dy / 2, self.ny)
            # Concatenate boundaries ([0] for 0 side, [1] for far side) and internal
            x = np.concatenate(([0.0], x_internal, [1.0]))
            y = np.concatenate(([0.0], y_internal, [1.0]))
            # Make into a grid
            self.x, self.y = np.meshgrid(x, y)


        ''' elif self.problem_type == 'cavity_step':
            self.Lx = 1.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx)
            self.dy = self.Ly / (self.ny)

            # Mesh Creation
            # Generate internal cell centers
            x_internal = np.linspace(self.dx / 2, 1 - self.dx / 2, self.nx)
            y_internal = np.linspace(self.dy / 2, 1 - self.dy / 2, self.ny)
            # Concatenate boundaries ([0] for 0 side, [1] for far side) and internal
            x = np.concatenate(([0.0], x_internal, [1.0]))
            y = np.concatenate(([0.0], y_internal, [1.0]))
            # Make into a grid
            self.x, self.y = np.meshgrid(x, y)

            step_x = self.nx // 3
            step_y = self.ny // 3
            self.solid_mask[:step_y, :step_x] = True

        elif self.problem_type == 'backstep':
            self.Lx = 2.0
            self.Ly = 0.25
            self.dx = self.Lx / (self.nx)
            self.dy = self.Lx / (self.ny)

            # Mesh Creation
            # Generate internal cell centers
            x_internal = np.linspace(self.dx / 2, 1 - self.dx / 2, self.nx)
            y_internal = np.linspace(self.dy / 2, 1 - self.dy / 2, self.ny)
            # Concatenate boundaries ([0] for 0 side, [1] for far side) and internal
            x = np.concatenate(([0.0], x_internal, [1.0]))
            y = np.concatenate(([0.0], y_internal, [1.0]))
            # Make into a grid
            self.x, self.y = np.meshgrid(x, y)

            step_x = self.nx // 2
            step_y = self.ny // 2
            self.solid_mask[:step_y, :step_x] = True '''

    def _apply_bc(self):
        # 1st: Velocity BCs

        # Apply no-slip to all walls by default (u=0, v=0)
        self.u[0,:] = 0.0
        self.u[-1,:] = 0.0
        self.u[:,0] = 0.0
        self.u[:,-1] = 0.0
        self.v[0,:] = 0.0
        self.v[-1,:] = 0.0
        self.v[:,0] = 0.0
        self.v[:,-1] = 0.0

        # Now, problem specific BCs
        if self.problem_type == 'cavity' or self.problem_type == 'cavity_step':
            self.u[0, 1:self.nx + 1] = 1.0
            self.u_star[0, 1:self.nx + 1] = 1.0
            self.u_face[0, 1:self.nx] = 1.0

        # if self.problem_type == 'backstep': # NO PARABOLIC PROFILE
        #     j_mid = self.ny // 2
        #     self.u[j_mid:,0] = 1.0
        #     self.u[:, -1] = self.u[:,-2] # Zero Gradient at outlet (approx)
         #    self.v[:,-1] = self.v[:,-2]

        # self.u[self.solid_mask] = 0.0
        # self.v[self.solid_mask] = 0.0

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
        # In below loops, i represents y and j represents x. Makes the [i,j] make sense
        # Look only at the interior nodes. Not the boundaries
        for i in range(2, self.ny):
            for j in range(2, self.nx):
                # Convective mass flux coefficients
                Fe, Fw, Fn, Fs = self._compute_convective_mass(i, j)
                # Convective Coefficients, Equation 15a-15e
                self.a_e[i,j] = De + max(0.0, -Fe)
                self.a_w[i,j] = Dw + max(0.0, Fw)
                self.a_n[i,j] = Dn + max(0.0, -Fn)
                self.a_s[i,j] = Ds + max(0.0, Fs)
                self.a_p[i,j] = self.a_e[i,j] + self.a_w[i,j] + self.a_n[i,j] + self.a_s[i,j] + (Fe - Fw) + (Fn - Fs)
                # Source Terms - Equations 17, 18
                self.b_u[i,j] = 0.5 * (self.p[i, j - 1] - self.p[i, j + 1]) * self.dy
                self.b_v[i,j] = 0.5 * (self.p[i + 1, j] - self.p[i - 1, j]) * self.dx

        # Now look at the Left Wall only. No corners
        j_l = 1
        for i_l in range (2, self.ny):
            # Convective mass flux coefficients
            Fe, Fw, Fn, Fs = self._compute_convective_mass(i_l, j_l)
            # Convective Coefficients, Equations 15a-15e. Modified as edge case
            self.a_e[i_l,j_l] = De + max(0.0, -Fe)
            self.a_w[i_l,j_l] = 2 * Dw + max(0.0, Fw)
            self.a_n[i_l,j_l] = Dn + max(0.0, -Fn)
            self.a_s[i_l,j_l] = Ds + max(0.0, Fs)
            self.a_p[i_l,j_l] = self.a_e[i_l,j_l] + self.a_w[i_l,j_l] + self.a_n[i_l,j_l] + self.a_s[i_l,j_l] + (Fe - Fw) + (Fn - Fs)
            # Source Terms - Equations 17, 18. Modified as edge case
            self.b_u[i_l, j_l] = 0.5 * (self.p[i_l, j_l] - self.p[i_l, j_l + 1]) * self.dy
            self.b_v[i_l, j_l] = 0.5 * (self.p[i_l + 1, j_l] - self.p[i_l - 1, j_l]) * self.dx

        # Bottom Wall. No corners
        i_b = self.ny
        for j_b in range(2, self.nx):
            # Convective mass flux coefficients
            Fe, Fw, Fn, Fs = self._compute_convective_mass(i_b, j_b)
            # Convective Coefficients, Equations 15a-15e. Modified as edge case
            self.a_e[i_b,j_b] = De + max(0.0, -Fe)
            self.a_w[i_b,j_b] = Dw + max(0.0, Fw)
            self.a_n[i_b,j_b] = Dn + max(0.0, -Fn)
            self.a_s[i_b,j_b] = 2 * Ds + max(0.0, Fs)
            self.a_p[i_b,j_b] = self.a_e[i_b,j_b] + self.a_w[i_b,j_b] + self.a_n[i_b,j_b] + self.a_s[i_b,j_b] + (Fe - Fw) + (Fn - Fs)
            # Source Terms - Equation 17, 18. Modified as edge case
            self.b_u[i_b,j_b] = 0.5 * (self.p[i_b,j_b - 1] - self.p[i_b, j_b + 1]) * self.dy
            self.b_v[i_b,j_b] = 0.5 * (self.p[i_b,j_b] - self.p[i_b - 1, j_b]) * self.dx

        # Right Wall. No Corners
        j_r = self.nx
        for i_r in range(2, self.ny):
            # Convective mass flux coefficients
            Fe, Fw, Fn, Fs = self._compute_convective_mass(i_r, j_r)
            # Convective Coefficients, Equations 15a - 15e. Modified as edge case
            self.a_e[i_r,j_r] = De + max(0.0, -Fe)
            self.a_w[i_r,j_r] = 2 * Dw + max(0.0, Fw)
            self.a_n[i_r,j_r] = Dn + max(0.0, -Fn)
            self.a_s[i_r,j_r] = Ds + max(0.0, Fs)
            self.a_p[i_r, j_r] = self.a_e[i_r, j_r] + self.a_w[i_r, j_r] + self.a_n[i_r, j_r] + self.a_s[i_r, j_r] + (Fe - Fw) + (Fn - Fs)
            # Source Terms - Equation 17, 18. Modified as edge case
            self.b_u[i_r, j_r] = 0.5 * (self.p[i_r, j_r - 1] - self.p[i_r, j_r]) * self.dy
            self.b_v[i_r, j_r] = 0.5 * (self.p[i_r + 1, j_r] - self.p[i_r - 1, j_r]) * self.dx

        # Top Wall. No Corners.
        i_t = 1
        for j_t in range(2, self.ny):
            # Convective mass flux
            Fe, Fw, Fn, Fs = self._compute_convective_mass(i_t, j_t)
            # Convective Coefficients, Equations 15a - 15e. Modified for edge cases
            self.a_e[i_t, j_t] = De + max(0.0, -Fe)
            self.a_w[i_t, j_t] = Dw + max(0.0, Fw)
            self.a_n[i_t, j_t] = 2 * Dn + max(0.0, -Fn)
            self.a_s[i_t, j_t] = Ds + max(0.0, Fs)
            self.a_p[i_t, j_t] = self.a_e[i_t, j_t] + self.a_w[i_t, j_t] + self.a_n[i_t, j_t] + self.a_s[i_t, j_t] + (Fe - Fw) + (Fn - Fs)
            # Source Terms - Equation 17, 18
            self.b_u[i_t, j_t] = 0.5 * (self.p[i_t, j_t - 1] - self.p[i_t, j_t + 1]) * self.dy
            self.b_v[i_t, j_t] = 0.5 * (self.p[i_t + 1, j_t] - self.p[i_t, j_t]) * self.dx

        # Top Left Corner
        i_tl = 1
        j_tl = 1
        # Convective mass flux
        Fe, Fw, Fn, Fs = self._compute_convective_mass(i_tl, j_tl)
        # Convective Coefficients
        self.a_e[i_tl, j_tl] = De + max(0.0, -Fe)
        self.a_w[i_tl, j_tl] = 2 * Dw + max(0.0, Fw)
        self.a_n[i_tl, j_tl] = 2 * Dn + max(0.0, -Fn)
        self.a_s[i_tl, j_tl] = Ds + max(0.0, Fs)
        self.a_p[i_tl, j_tl] = self.a_e[i_tl, j_tl] + self.a_w[i_tl, j_tl] + self.a_n[i_tl, j_tl] + self.a_s[i_tl, j_tl] + (Fe - Fw) + (Fn - Fs)
        # Source Terms - Equation 17, 18
        self.b_u[i_tl, j_tl] = 0.5 * (self.p[i_tl, j_tl] - self.p[i_tl, j_tl + 1]) * self.dy
        self.b_v[i_tl, j_tl] = 0.5 * (self.p[i_tl + 1, j_tl] - self.p[i_tl, j_tl]) * self.dx

        # Top Right Corner
        i_tr = 1
        j_tr = self.nx
        # Convective mass flux
        Fe, Fw, Fn, Fs = self._compute_convective_mass(i_tr, j_tr)
        # Convective Coefficients
        self.a_e[i_tr, j_tr] = De + max(0.0, -Fe)
        self.a_w[i_tr, j_tr] = 2 * Dw + max(0.0, Fw)
        self.a_n[i_tr, j_tr] = 2 * Dn + max(0.0, -Fn)
        self.a_s[i_tr, j_tr] = Ds + max(0.0, Fs)
        self.a_p[i_tr, j_tr] = self.a_e[i_tr, j_tr] + self.a_w[i_tr, j_tr] + self.a_n[i_tr, j_tr] + self.a_s[i_tr, j_tr] + (Fe - Fw) + (Fn - Fs)
        # Source Terms - Equation 17, 18
        self.b_u[i_tr, j_tr] = 0.5 * (self.p[i_tr, j_tr - 1] - self.p[i_tr, j_tr]) * self.dy
        self.b_v[i_tr, j_tr] = 0.5 * (self.p[i_tr + 1, j_tr] - self.p[i_tr, j_tr]) * self.dx

        # Bottom Left Corner
        i_bl = self.ny
        j_bl = 1
        # Convective mass flux
        Fe, Fw, Fn, Fs = self._compute_convective_mass(i_bl, j_bl)
        # Convective Coefficients
        self.a_e[i_bl, j_bl] = De + max(0.0, -Fe)
        self.a_w[i_bl, j_bl] = 2 * Dw + max(0.0, Fw)
        self.a_n[i_bl, j_bl] = 2 * Dn + max(0.0, -Fn)
        self.a_s[i_bl, j_bl] = Ds + max(0.0, Fs)
        self.a_p[i_bl, j_bl] = self.a_e[i_bl, j_bl] + self.a_w[i_bl, j_bl] + self.a_n[i_bl, j_bl] + self.a_s[i_bl, j_bl] + (Fe - Fw) + (Fn - Fs)
        # Source Terms - Equation 17, 18
        self.b_u[i_bl, j_bl] = 0.5 * (self.p[i_bl, j_bl] - self.p[i_bl, j_bl + 1]) * self.dy
        self.b_v[i_bl, j_bl] = 0.5 * (self.p[i_bl, j_bl] - self.p[i_bl - 1, j_bl]) * self.dx

        # Bottom Right Corner
        i_br = self.ny
        j_br = self.nx
        # Convective mass flux
        Fe, Fw, Fn, Fs = self._compute_convective_mass(i_br, j_br)
        # Convective Coefficients
        self.a_e[i_br, j_br] = De + max(0.0, -Fe)
        self.a_w[i_br, j_br] = 2 * Dw + max(0.0, Fw)
        self.a_n[i_br, j_br] = 2 * Dn + max(0.0, -Fn)
        self.a_s[i_br, j_br] = Ds + max(0.0, Fs)
        self.a_p[i_br, j_br] = self.a_e[i_br, j_br] + self.a_w[i_br, j_br] + self.a_n[i_br, j_tl] + self.a_s[i_br, j_tl] + (Fe - Fw) + (Fn - Fs)
        # Source Terms - Equation 17, 18
        self.b_u[i_br, j_br] = 0.5 * (self.p[i_br, j_br] - self.p[i_br, j_br + 1]) * self.dy
        self.b_v[i_br, j_br] = 0.5 * (self.p[i_br, j_br] - self.p[i_br - 1, j_br]) * self.dx


    def _solve_uv_matrix(self):
        for n_u in range(1, self.max_iterations + 1):
            error_u = 0
            for i in range(1, self.ny + 1):
                for j in range(1, self.nx + 1):
                    self.u[i,j] = self.alpha_uv * (
                            self.a_e[i,j] * self.u[i, j + 1] +
                            self.a_w[i,j] * self.u[i,j-1] +
                            self.a_n[i,j] * self.u[i-1,j] +
                            self.a_s[i,j] * self.u[i+1,j] +
                            self.b_u[i,j]) / self.a_p[i,j] + (1 - self.alpha_uv) * self.u_star[i,j]
                    error_u += (self.u[i,j] - self.alpha_uv * (
                            self.a_e[i, j] * self.u[i, j + 1] +
                            self.a_w[i, j] * self.u[i, j - 1] +
                            self.a_n[i, j] * self.u[i - 1, j] +
                            self.a_s[i, j] * self.u[i + 1, j] +
                            self.b_u[i, j]) / self.a_p[i, j] + (1 - self.alpha_uv) * self.u_star[i, j])**2

            if n_u == 1: # Outer Iteration Metric for SIMPLE
                norm_u = math.sqrt(error_u)
            error_u = math.sqrt(error_u)

            if error_u < self.tolerance:
                # Inner iteration residual has converged (Gauss-Seidel iterations)
                break

        for n_v in range(1, max_inner_iteration_uv + 1):
            error_v = 0
            for i in range(1, self.ny + 1):
                for j in range(1, self.nx + 1):
                    self.v[i, j] = self.alpha_uv * (
                            self.a_e[i, j] * self.v[i, j + 1] +
                            self.a_w[i, j] * self.v[i, j - 1] +
                            self.a_n[i, j] * self.v[i - 1, j] +
                            self.a_s[i, j] * self.v[i + 1, j] +
                            self.b_v[i, j]) / self.a_p[i, j] + (1 - self.alpha_uv) * self.v_star[i, j]
                    error_v += (self.v[i, j] - self.alpha_uv * (
                            self.a_e[i, j] * self.v[i, j + 1] +
                            self.a_w[i, j] * self.v[i, j - 1] +
                            self.a_n[i, j] * self.v[i - 1, j] +
                            self.a_s[i, j] * self.v[i + 1, j] +
                            self.b_v[i, j]) / self.a_p[i, j] + (1 - self.alpha_uv) * self.v_star[i, j]) ** 2
            if n_v == 1:
                norm_v = math.sqrt(error_v)
            error_v = math.sqrt(error_v)
            if error_v < self.tolerance:
                break

        return norm_u, norm_v

    def pressure_correction(self):
        # start with interior cells only.
        for i in range(2, self.ny):
            for j in range(2, self.nx):
                self.Ap_e[i,j] = 0.5 * self.alpha_uv * (1 / self.a_p[i,j] + 1 / self.a_p[i,j+1]) * (self.dy **2)
                self.Ap_w[i,j] = 0.5 * self.alpha_uv * (1 / self.a_p[i,j] + 1 / self.a_p[i,j-1]) * (self.dy **2)
                self.Ap_n[i,j] = 0.5 * self.alpha_uv * (1 / self.a_p[i,j] + 1 / self.a_p[i-1,j]) * (self.dx **2)
                self.Ap_s[i,j] = 0.5 * self.alpha_uv * (1 / self.a_p[i,j] + 1 / self.a_p[i+1,j]) * (self.dx **2)
                self.Ap_p[i,j] = self.Ap_e[i,j] + self.Ap_w[i,j] + self.Ap_n[i,j] + self.Ap_s[i,j] + self.Ap_s[i,j]

                self.b_p[i,j] = -(self.u_face[i,j] - self.u_face[i,j-1]) * self.dy - (self.v_face[i-1,j] - self.v_face[i,j]) * self.dx

        # top boundary. no corners
        i_t = 1
        for j_t in range(2, self.nx):
            self.Ap_e[i_t, j_t] = 0.5 * self.alpha_uv * (1 / self.a_p[i_t, j_t] + 1 / self.a_p[i_t, j_t + 1]) * (self.dy ** 2)
            self.Ap_w[i_t, j_t] = 0.5 * self.alpha_uv * (1 / self.a_p[i_t, j_t] + 1 / self.a_p[i_t, j_t - 1]) * (self.dy ** 2)
            self.Ap_n[i_t, j_t] = 0
            self.Ap_s[i_t, j_t] = 0.5 * self.alpha_uv * (1 / self.a_p[i_t, j_t] + 1 / self.a_p[i_t + 1, j_t]) * (self.dx ** 2)
            self.Ap_p[i_t, j_t] = self.Ap_e[i_t, j_t] + self.Ap_w[i_t, j_t] + self.Ap_n[i_t, j_t] + self.Ap_s[i_t, j_t] + self.Ap_s[i_t, j_t]

            self.b_p[i_t, j_t] = -(self.u_face[i_t, j_t] - self.u_face[i_t, j_t - 1]) * self.dy - (self.v_face[i_t - 1, j_t] - self.v_face[i_t, j_t]) * self.dx

        # left boundary. no corners
        j_l = 1
        for i_l in range(2, self.ny):
            self.Ap_e[i_l, j_l] = 0.5 * self.alpha_uv * (1 / self.a_p[i_l, j_l] + 1 / self.a_p[i_l, j_l + 1]) * (self.dy ** 2)
            self.Ap_w[i_l, j_l] = 0
            self.Ap_n[i_l, j_l] = 0.5 * self.alpha_uv * (1 / self.a_p[i_l, j_l] + 1 / self.a_p[i_l - 1, j_l]) * (self.dx ** 2)
            self.Ap_s[i_l, j_l] = 0.5 * self.alpha_uv * (1 / self.a_p[i_l, j_l] + 1 / self.a_p[i_l + 1, j_l]) * (self.dx ** 2)
            self.Ap_p[i_l, j_l] = self.Ap_e[i_l, j_l] + self.Ap_w[i_l, j_l] + self.Ap_n[i_l, j_l] + self.Ap_s[i_l, j_l] + self.Ap_s[i_l, j_l]

            self.b_p[i_l, j_l] = -(self.u_face[i_l, j_l] - self.u_face[i_l, j_l - 1]) * self.dy - (self.v_face[i_l - 1, j_l] - self.v_face[i_l, j_l]) * self.dx

        # right boundary. no corners
        j_r = self.nx
        for i_r in range(2, self.ny):
            self.Ap_e[i_r, j_r] = 0
            self.Ap_w[i_r, j_r] = 0.5 * self.alpha_uv * (1 / self.a_p[i_r, j_r] + 1 / self.a_p[i_r, j_r + 1]) * (self.dy ** 2)
            self.Ap_n[i_r, j_r] = 0.5 * self.alpha_uv * (1 / self.a_p[i_r, j_r] + 1 / self.a_p[i_r - 1, j_r]) * (self.dx ** 2)
            self.Ap_s[i_r, j_r] = 0.5 * self.alpha_uv * (1 / self.a_p[i_r, j_r] + 1 / self.a_p[i_r + 1, j_r]) * (self.dx ** 2)
            self.Ap_p[i_r, j_r] = self.Ap_e[i_r, j_r] + self.Ap_w[i_r, j_r] + self.Ap_n[i_r, j_r] + self.Ap_s[i_r, j_r] + self.Ap_s[i_r, j_r]

            self.b_p[i_r, j_r] = -(self.u_face[i_r, j_r] - self.u_face[i_r, j_r - 1]) * self.dy - (self.v_face[i_r - 1, j_r] - self.v_face[i_r, j_r]) * self.dx

        # bottom boundary. no corners
        i_b = self.ny
        for j_b in range(2, self.nx):
            self.Ap_e[i_b, j_b] = 0.5 * self.alpha_uv * (1 / self.a_p[i_b, j_b] + 1 / self.a_p[i_b, j_b + 1]) * (self.dy ** 2)
            self.Ap_w[i_b, j_b] = 0.5 * self.alpha_uv * (1 / self.a_p[i_b, j_b] + 1 / self.a_p[i_b, j_b + 1]) * (self.dy ** 2)
            self.Ap_n[i_b, j_b] = 0.5 * self.alpha_uv * (1 / self.a_p[i_b, j_b] + 1 / self.a_p[i_b - 1, j_b]) * (self.dx ** 2)
            self.Ap_s[i_b, j_b] = 0
            self.Ap_p[i_b, j_b] = self.Ap_e[i_b, j_b] + self.Ap_w[i_b, j_b] + self.Ap_n[i_b, j_b] + self.Ap_s[i_b, j_b] + self.Ap_s[i_b, j_b]

            self.b_p[i_b, j_b] = -(self.u_face[i_b, j_b] - self.u_face[i_b, j_b - 1]) * self.dy - (self.v_face[i_b - 1, j_b] - self.v_face[i_b, j_b]) * self.dx

    def correct_face_velocity(self):
        for i in range(1, self.ny + 1):
            for j in range(1, self.nx):
                self.u_face[i,j] = self.u_face[i,j] + 0.5 * self.alpha_uv * (1 / self.a_p[i,j] + 1 / self.a_p[i,j+1]) * (self.p_prime[i,j] - self.p_prime[i, j+1]) * self.dy

        for i in range(2, self.ny + 1):
            for j in range (1, self.nx + 1):
                self.v_face[i-1,j] = self.v_face[i-1,j] + 0.5 * self.alpha_uv * (1 / self.a_p[i,j] + 1 / self.a_p[i-1,j]) * (self.p_prime[i,j] - self.p_prime[i-1,j]) * self.dx

    def SIMPLE(self):
        for n_simple in range(1, self.max_iterations + 1):
            self._solve_momentum()
            error_u, error_v = self._solve_uv_matrix()
            print(f'Iteration {n_simple}: u:{error_u:.4f}, v:{error_v:.4f}')
            self._correct_face_velocity


if __name__ == "__main__":


    cavitySolve = solver(nx = 33, ny = 33, Re = 100, problem_type = 'cavity')
    cavitySolve.SIMPLE()

    #cavityStepSolve = solver(nx=33, ny=33, Re=100, problem_type='cavity_step')
    # backstepSolve = solver(nx = 33, ny=33, Re=100, problem_type='backstep')




    # Actual Assignment Cases:
    # cavitySolve = solver(257, 257, 100, 'cavity')
    # cavityStepSolve = solver(320, 320, 200, 'cavity_step'
    # backstepSolve = solver(320, 160, 100, 'backstep')
    # backstepSolve = solver(320, 160, 200, 'backstep')
    # backstepSolve = solver(320, 160, 400, 'backstep')



