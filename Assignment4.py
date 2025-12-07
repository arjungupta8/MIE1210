import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class Solver:
    def __init__(self, nx: int, ny: int, Re: float, problem_type: str):
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.problem_type = problem_type

        self._setup_geometry()

        self.u = np.zeros((ny, nx))  # x vel
        self.v = np.zeros((ny, nx)) # y vel
        self.p = np.zeros((ny, nx)) # pressure

        self.u_star = np.zeros((ny, nx))
        self.v_star = np.zeros((ny, nx))
        self.p_prime = np.zeros((ny, nx))

        self.d_u = np.ones((ny, nx)) * 1e-10 # Rhie-Chow coefficient for x vel. Small +ve val to avoid division errors
        self.d_v = np.ones((ny, nx)) * 1e-10 # Rhie-Chow coefficient for y vel

        self.alpha_u = 0.5 # Relaxation in x vel
        self.alpha_v = 0.5 # Relaxation in y vel
        self.alpha_p = 0.2 # Relaxation in pressure


    def _setup_geometry(self):
        if self.problem_type == 'cavity':
            self.Lx = 1.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx-1) # mesh size in x
            self.dy = self.Ly / (self.ny-1) # mesh size in y
            self.x = np.linspace(0, self.Lx, self.nx)
            self.y = np.linspace(0, self.Ly, self.ny)
            self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool) # Shows the nodes that has a physical obstacle

        elif self.problem_type == 'cavity_step':
            self.Lx = 1.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx-1)
            self.dy = self.Ly / (self.ny-1)
            self.x = np.linspace(0, self.Lx, self.nx)
            self.y = np.linspace(0, self.Ly, self.ny)
            self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool)
            step_x = self.nx // 4
            step_y = self.ny // 4
            self.solid_mask[:step_y, :step_x] = True # as bottom left corner near 0,0 has the step

        elif self.problem_type == 'backstep':
            self.Lx = 2.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx-1)
            self.dy = self.Ly / (self.ny-1)
            self.x = np.linspace(0, self.Lx, self.nx)
            self.y = np.linspace(0, self.Ly, self.ny)
            self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool)
            step_y = self.ny // 2
            self.solid_mask[:step_y, 0] = True # matches the geometry

    def _compute_pressure_gradient(self, field, j, i, direction):
        if direction == 'x':
            if i == 0:
                return (field[j,i+1] - field[j, i]) / self.dx
            elif i == self.nx - 1:
                return (field[j, i] - field[j, i-1]) / self.dx
            else:
                return (field[j, i+1] - field[j, i-1]) / (2 * self.dx)
        elif direction == 'y':
            if j == 0:
                return (field[j+1, i] - field[j, i]) / self.dy
            elif j == self.ny - 1:
                return (field[j, i] - field[j-1, i]) / self.dy
            else:
                return (field[j+1, i] - field[j-1, i]) / (2 * self.dy)

    def _rhie_chow_u_face(self, j, i, face):
        if face == 'east':
            if i >= self.nx - 1: # Far east side
                return self.u[j, i]

            u_avg = 0.5 * (self.u[j, i] + self.u[j, i+1]) # Avg of velocities
            d_avg = 0.5 * (self.d_u[j, i] + self.d_u[j, i+1]) # average of d coefficients
            dp_dx_face = (self.p[j, i+1] - self.p[j, i]) / self.dx # Central Difference pressure gradient across face
            dp_dx_P = self._compute_pressure_gradient(self.p, j, i, 'x')
            dp_dx_E = self._compute_pressure_gradient(self.p, j, i+1, 'x')
            u_face = u_avg - d_avg * (dp_dx_face - 0.5 * (dp_dx_P + dp_dx_E)) # Equation 32 in notes, probably different in report

            return u_face

        elif face == 'west':
            if i <= 0:
                return self.u[j, i]

            u_avg = 0.5 * (self.u[j, i] + self.u[j, i-1])
            d_avg = 0.5 * (self.d_u[j, i] + self.d_u[j, i-1])
            dp_dx_face = (self.p[j, i] - self.p[j, i-1]) / self.dx
            dp_dx_P = self._compute_pressure_gradient(self.p, j, i, 'x')
            dp_dx_W = self._compute_pressure_gradient(self.p, j, i-1, 'x')
            u_face = u_avg - d_avg * (dp_dx_face - 0.5 * (dp_dx_P + dp_dx_W))

            return u_face

        elif face == 'north':
            if j >= self.ny - 1:
                return self.u[j, i]

            u_avg = 0.5 * (self.u[j, i] + self.u[j+1, i])
            d_avg = 0.5 * (self.d_u[j, i] + self.d_u[j+1, i])
            dp_dy_face = (self.p[j+1, i] - self.p[j, i]) / self.dy
            dp_dy_P = self._compute_pressure_gradient(self.p, j, i, 'y')
            dp_dy_N = self._compute_pressure_gradient(self.p, j+1, i, 'y')
            u_face = u_avg - d_avg * (dp_dy_face - 0.5 * (dp_dy_P + dp_dy_N))

            return u_face

        elif face == 'south':
            if j <= 0:
                return self.u[j, i]

            u_avg = 0.5* (self.u[j, i] + self.u[j-1, i])
            d_avg = 0.5* (self.d_u[j, i] + self.d_u[j-1, i])
            dp_dy_face = (self.p[j, i] - self.p[j-1, i]) / self.dy
            dp_dy_P = self._compute_pressure_gradient(self.p, j, i, 'y')
            dp_dy_S = self._compute_pressure_gradient(self.p, j-1, i, 'y')
            u_face = u_avg - d_avg * (dp_dy_face - 0.5 * (dp_dy_P + dp_dy_S))

            return u_face

    def _rhie_chow_v_face(self, j, i, face):
        if face == 'north':
            if j >= self.ny - 1:
                return self.v[j, i]

            v_avg = 0.5 * (self.v[j, i] + self.v[j + 1, i])
            d_avg = 0.5 * (self.d_v[j, i] + self.d_v[j + 1, i])
            dp_dy_face = (self.p[j + 1, i] - self.p[j, i]) / self.dy
            dp_dy_P = self._compute_pressure_gradient(self.p, j, i, 'y')
            dp_dy_N = self._compute_pressure_gradient(self.p, j + 1, i, 'y')
            v_face = v_avg - d_avg * (dp_dy_face - 0.5 * (dp_dy_P + dp_dy_N))

            return v_face

        elif face == 'south':
            if j <= 0:
                return self.v[j, i]

            v_avg = 0.5 * (self.v[j, i] + self.v[j - 1, i])
            d_avg = 0.5 * (self.d_v[j, i] + self.d_v[j - 1, i])
            dp_dy_face = (self.p[j, i] - self.p[j - 1, i]) / self.dy
            dp_dy_P = self._compute_pressure_gradient(self.p, j, i, 'y')
            dp_dy_S = self._compute_pressure_gradient(self.p, j - 1, i, 'y')
            v_face = v_avg - d_avg * (dp_dy_face - 0.5 * (dp_dy_P + dp_dy_S))

            return v_face

        elif face == 'east':
            if i >= self.nx - 1:
                return self.v[j, i]

            v_avg = 0.5 * (self.v[j, i] + self.v[j, i + 1])
            d_avg = 0.5 * (self.d_v[j, i] + self.d_v[j, i + 1])
            dp_dx_face = (self.p[j, i + 1] - self.p[j, i]) / self.dx
            dp_dx_P = self._compute_pressure_gradient(self.p, j, i, 'x')
            dp_dx_E = self._compute_pressure_gradient(self.p, j, i + 1, 'x')
            v_face = v_avg - d_avg * (dp_dx_face - 0.5 * (dp_dx_P + dp_dx_E))

            return v_face

        elif face == 'west':
            if i <= 0:
                return self.v[j, i]

            v_avg = 0.5 * (self.v[j, i] + self.v[j, i - 1])
            d_avg = 0.5 * (self.d_v[j, i] + self.d_v[j, i - 1])
            dp_dx_face = (self.p[j, i] - self.p[j, i - 1]) / self.dx
            dp_dx_P = self._compute_pressure_gradient(self.p, j, i, 'x')
            dp_dx_W = self._compute_pressure_gradient(self.p, j, i - 1, 'x')
            v_face = v_avg - d_avg * (dp_dx_face - 0.5 * (dp_dx_P + dp_dx_W))

            return v_face

    def _solve_momentum_u(self):
        n = self.nx * self.ny # number of nodes, for A and b matrix sizes
        A = lil_matrix((n, n)) # LIL = List of Lists. Effecient construction of sparse matrices
        b = np.zeros(n)

        nu = 1.0/self.Re

        #First set the BC in matrix
        for j in range (self.ny):
            for i in range (self.nx):
                idx = j * self.nx + i
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1 or self.solid_mask[j, i]:
                    A[idx, idx] = 1.0
                    b[idx] = self.u[j, i]

        # Solve interior nodes
        for j in range (1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if self.solid_mask[j, i]:
                    continue
                idx = j * self.nx + i
                # Diffusion flux coefficient, equation 11 in notes
                De = nu * self.dy / self.dx
                Dw = nu * self.dy / self.dx
                Dn = nu * self.dx / self.dy
                Ds = nu * self.dx / self.dy
                # Convective flux coefficients, found with Rhie-Chow Interpolation
                ue = self._rhie_chow_u_face(j, i, 'east')
                uw = self._rhie_chow_u_face(j, i, 'west')
                vn = self._rhie_chow_v_face(j, i, 'north')
                vs = self._rhie_chow_v_face(j, i, 'south')
                # Mass Fluxes
                Fe = ue * self.dy
                Fw = uw * self.dy
                Fn = vn * self.dx
                Fs = vs * self.dx
                # Convective Coefficients, Upwind Scheme. Equations 13a-d in notes
                aE = De + max(-Fe, 0)
                aW = Dw + max(Fw, 0)
                aN = Dn + max(-Fn, 0)
                aS = Ds + max(Fs, 0)

                aP = aE + aW + aN + aS + (Fe-Fw) + (Fn - Fs) # Equation 13e in notes
                if aP < 1e-8: aP = 1e-8
                self.d_u[j, i] = self.dx * self.dy / aP if aP != 0 else 1e-12 # Store d_u coefficients for R-C before under-relaxation
                aP_relaxed = aP / self.alpha_u # apply under-relaxation
                dp_dx = (self.p[j, i+1] - self.p[j, i-1]) / (2* self.dx) # pressure gradient

                # Build matrix
                A[idx, idx] = aP_relaxed
                if i < self.nx-1 and not self.solid_mask[j, i+1]:
                    A[idx, idx+1] = -aE
                if i > 0 and not self.solid_mask[j, i-1]:
                    A[idx, idx-1] = -aW
                if j< self.ny - 1 and not self.solid_mask[j+1, i]:
                    A[idx, idx+self.nx] = -aN
                if j > 0 and not self.solid_mask[j-1, i]:
                    A[idx, idx-self.nx] = -aS

                b[idx] = -dp_dx * self.dx * self.dy + (1 - self.alpha_u) * aP_relaxed * self.u[j,i] # Equation 14

        A = A.tocsr()
        u_flat = spsolve(A,b) # can change this to an iterative solver if required
        self.u_star = u_flat.reshape((self.ny, self.nx))


    def _solve_momentum_v(self):
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)
        nu = 1.0/self.Re

        for j in range(self.ny):  # Apply BCs to matrix
            for i in range(self.nx):
                idx = j * self.nx + i
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1 or self.solid_mask[j, i]:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = self.v[j, i]

        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):
                if self.solid_mask[j, i]:
                    continue

                idx = j * self.nx + i
                # Diffusion flux coefficient, equation 11 in notes
                De = nu * self.dy / self.dx
                Dw = nu * self.dy / self.dx
                Dn = nu * self.dx / self.dy
                Ds = nu * self.dx / self.dy
                # Convective flux coefficients, found with Rhie-Chow Interpolation
                ue = self._rhie_chow_u_face(j, i, 'east')
                uw = self._rhie_chow_u_face(j, i, 'west')
                vn = self._rhie_chow_v_face(j, i, 'north')
                vs = self._rhie_chow_v_face(j, i, 'south')
                # Mass Fluxes
                Fe = ue * self.dy
                Fw = uw * self.dy
                Fn = vn * self.dx
                Fs = vs * self.dx
                # Convective Coefficients, Upwind Scheme. Equations 13a-d in notes
                aE = De + max(-Fe, 0)
                aW = Dw + max(Fw, 0)
                aN = Dn + max(-Fn, 0)
                aS = Ds + max(Fs, 0)

                aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)  # Equation 13e in notes
                if aP < 1e-8: aP = 1e-8
                self.d_v[j, i] = self.dx * self.dy / aP if aP != 0 else 1e-12  # Store d_v coefficients for R-C before under-relaxation

                aP_relaxed = aP / self.alpha_v  # apply under-relaxation. Same as applying it later (equal to eqn 17 in handwritten)
                dp_dy = (self.p[j+1, i] - self.p[j-1, i]) / (2 * self.dy)  # pressure gradient

                # Build matrix
                A[idx, idx] = aP_relaxed
                if i < self.nx - 1 and not self.solid_mask[j, i+1]:
                    A[idx, idx + 1] = -aE
                if i > 0 and not self.solid_mask[j, i-1]:
                    A[idx, idx - 1] = -aW
                if j < self.ny - 1 and not self.solid_mask[j+1, i]:
                    A[idx, idx + self.nx] = -aN
                if j > 0 and not self.solid_mask[j-1, i]:
                    A[idx, idx - self.nx] = -aS

                b[idx] = -dp_dy * self.dx * self.dy + (1 - self.alpha_v) * aP_relaxed * self.v[j, i] # Equation 14


        A = A.tocsr()
        v_flat = spsolve(A,b) # can use iterative if required
        self.v_star = v_flat.reshape((self.ny, self.nx))

    def _solve_pressure_correction(self):
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)

        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):
                if self.solid_mask[j, i]:
                    continue
                idx = j * self.nx + i
                # Pressure Correction Coefficients, Equation 25, 26
                aE = self.d_u[j, i + 1] * self.dy # if i < self.nx - 2 else 0.0
                aW = self.d_u[j, i-1] * self.dy # if i > 1 else 0.0
                aN = self.d_v[j + 1, i] * self.dx # if j < self.ny-2 else 0.0
                aS = self.d_v[j - 1, i] * self.dx # if j > 1 else 0.0
                # aE = self.dy / self.dx
                # aW = self.dy / self.dx
                # aN = self.dx / self.dy
                # aS = self.dx / self.dy

                aP = aE + aW + aN + aS

                # Mass Source (Continuity violation). Equation 27, bracketed section
                div_u = ((self.u_star[j, i + 1] - self.u_star[j, i - 1]) / (2 * self.dx) +
                         (self.v_star[j + 1, i] - self.v_star[j - 1, i]) / (2 * self.dy))

                # u_e = 0.5 * (self.u_star[j, i] + self.u_star[j, i + 1])
                # u_w = 0.5 * (self.u_star[j, i - 1] + self.u_star[j, i])
                # v_n = 0.5 * (self.v_star[j, i] + self.v_star[j + 1, i])
                # v_s = 0.5 * (self.v_star[j - 1, i] + self.v_star[j, i])

                # Fe = u_e * self.dy
                # Fw = u_w * self.dy
                # Fn = v_n * self.dx
                # Fs = v_s * self.dx


                # b[idx] = (Fe - Fw + Fn - Fs)

                #Build Matrix from above equations
                A[idx, idx] = aP
                if i < self.nx-1:
                    A[idx, idx + 1] = -aE
                if i > 0:
                    A[idx, idx - 1] = -aW
                if j < self.ny - 1:
                    A[idx, idx + self.nx] = -aN
                if j > 0:
                    A[idx, idx - self.nx] = -aS
                b[idx] = div_u * self.dx * self.dy # Equation 45 in report.


        if self.problem_type == 'backstep':
            # Backstep uses mixed BC. Other two use Neumann everywhere (below)
            for j in range(self.ny): # Outlet - right side boundary uses Dirichlet
                idx =j * self.nx + (self.nx - 1)
                if not self.solid_mask[j, self.nx - 1]:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = 0 # Outlet pressure correction p' = 0

            for j in range(self.ny): # Inlet - Left side boundary uses Neumann
                idx = j * self.nx
                if not self.solid_mask[j, 0]:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    A[idx, idx + 1] = -1
                    b[idx] = 0

            for i in range(1, self.nx - 1): # Top Wall boundary uses Neumann. Skip Corners
                idx = (self.ny - 1) * self.nx + i
                A[idx, :] = 0
                A[idx, idx] = 1
                A[idx, idx - self.nx] = -1
                b[idx] = 0

            for i in range(1, self.nx - 1): # Bottom boundary uses Neumann, skips corners
                idx = i
                A[idx, :] = 0
                A[idx, idx] = 1
                A[idx, idx + self.nx] = -1
                b[idx] = 0

        else: # Other two problems, Neumann everywhere, one reference pressure
            for i in range(1, self.nx - 1): # Bottom Wall, skip corners
                idx = i
                A[idx, :] = 0
                A[idx, idx] = 1
                if idx + self.nx < n:
                    A[idx, idx + self.nx] = -1
                # if i > 0 and i < self.nx - 1:
                #     A[idx, idx -self.nx] = 1
                # A[idx, idx + self.nx] = -1
                b[idx] = 0

            for i in range(1, self.nx - 1): # Top Wall, skips corners
                idx = (self.ny - 1) * self.nx + i
                A[idx, :] = 0
                A[idx, idx] = 1
                # if i > 0 and i < self.nx - 1: #Interior Nodes
                #     A[idx, idx - self.nx] = -1 # p'[top] = p'[bottom]
                # A[idx, idx - self.nx] = -1
                if idx - self.nx >= 0:
                    A[idx, idx - self.nx] = -1
                b[idx] = 0

            for j in range(self.ny): # Left Wall, Neumann.
                idx = j * self.nx
                A[idx, :] = 0
                A[idx, idx] = 1
                if idx + 1 < n:
                    A[idx, idx + 1] = -1
                b[idx] = 0

            for j in range(self.ny): # Right Wall
                idx = j * self.nx + (self.nx - 1)
                A[idx, :] = 0
                A[idx, idx] = 1
                if idx - 1 >= 0:
                    A[idx, idx - 1] = -1 #p'[right] = p'[left]
                b[idx] = 0

            # Reference Pressure (set one point to zero)
            ref_idx = (self.ny // 2) * self.nx + (self.nx // 2)
            A[ref_idx, :] = 0
            A[ref_idx, ref_idx] = 1
            b[ref_idx] = 0

            # Change reference pressure from mid cell to 0,0
            # A[0,:] = 0
            # A[0, 0] = 1
            # b[0] = 0

        A = A.tocsr()
        p_prime_flat = spsolve(A, b)
        self.p_prime = p_prime_flat.reshape((self.ny, self.nx))


    def _apply_boundary_conditions(self):
        if self.problem_type == 'cavity':
            # Top wall moves with u=1
            self.u[-1, :] = 1.0
            self.v[-1, :] = 0.0

            # Bottom, left, right walls: no-slip
            self.u[0, :] = 0.0
            self.v[0, :] = 0.0
            self.u[:, 0] = 0.0
            self.v[:, 0] = 0.0
            self.u[:, -1] = 0.0
            self.v[:, -1] = 0.0

        elif self.problem_type == 'cavity_step':
            # Top wall (lid) moves with u=1
            self.u[-1, :] = 1.0
            self.v[-1, :] = 0.0

            # Bottom, left, right walls: no-slip
            self.u[0, :] = 0.0
            self.v[0, :] = 0.0
            self.u[:, 0] = 0.0
            self.v[:, 0] = 0.0
            self.u[:, -1] = 0.0
            self.v[:, -1] = 0.0

            # Solid mask for step
            self.u[self.solid_mask] = 0.0
            self.v[self.solid_mask] = 0.0

        elif self.problem_type == 'backstep':
            # Inlet (parabolic profile on upper half)
            step_y = self.ny // 2
            for j in range(step_y, self.ny):
                y_local = (j - step_y) * self.dy
                h = (self.ny - step_y) * self.dy
                self.u[j, 0] = 1.5 * (1 - ((y_local - h / 2) / (h / 2)) ** 2)
            self.v[:, 0] = 0.0

            # Outlet: zero gradient
            self.u[:, -1] = self.u[:, -2]
            self.v[:, -1] = self.v[:, -2]

            # Walls: no-slip
            self.u[0, :] = 0.0
            self.v[0, :] = 0.0
            self.u[-1, :] = 0.0
            self.v[-1, :] = 0.0

    def _correct_velocity_pressure(self):
        # Update pressure
        self.p += self.alpha_p * self.p_prime

        # Update velocities
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if not self.solid_mask[j, i]:
                    dp_dx = (self.p_prime[j, i + 1] - self.p_prime[j, i - 1]) / (2 * self.dx)
                    dp_dy = (self.p_prime[j + 1, i] - self.p_prime[j - 1, i]) / (2 * self.dy)

                    self.u[j, i] = self.u_star[j, i] - self.d_u[j, i] * dp_dx
                    self.v[j, i] = self.v_star[j, i] - self.d_v[j, i] * dp_dy
        self._apply_boundary_conditions()

    def _compute_divergence(self):
        div = np.zeros((self.ny, self.nx))
        for j in range(1, self.ny - 1):
            for i in range (1, self.nx - 1):
                if not self.solid_mask[j, i]:
                    div[j, i] = ((self.u[j, i + 1] - self.u[j, i - 1]) / (2 * self.dx) +
                                 (self.v[j + 1, i] - self.v[j - 1, i]) / (2 * self.dy))
        return np.max(np.abs(div))

    def solve(self, max_iter = 1000, tolerance = 1e-6):
        print (f"Solving {self.problem_type} at Re={self.Re}")

        self._apply_boundary_conditions()


        for iteration in range (max_iter):
            u_old = self.u.copy() # u_star
            v_old = self.v.copy() # effectively v_star

            # SIMPLE steps
            self._solve_momentum_u()
            self._solve_momentum_v()
            self._solve_pressure_correction()
            self._correct_velocity_pressure()

            # Check convergence
            u_diff = np.max(np.abs(self.u - u_old))
            v_diff = np.max(np.abs(self.v - v_old))
            max_diff = max(u_diff, v_diff)

            if iteration % 5 == 0:
                div_max = self._compute_divergence()
                print(f"Iteration {iteration}: max_diff={max_diff:.6e}, max_div={div_max:.6e}")

            if max_diff < tolerance:
                print(f"Converged in {iteration} iterations")
                break

        return self.u, self.v, self.p

    def plot_results(self, save_prefix=''):
        X, Y = np.meshgrid(self.x, self.y)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # U Velocity
        im1 = axes[0, 0].contourf(X, Y, self.u, levels=20, cmap='RdBu_r')
        axes[0, 0].set_title('U Velocity')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 0])

        # V Velocity
        im2 = axes[0, 1].contourf(X, Y, self.v, levels=20, cmap='RdBu_r')
        axes[0, 1].set_title('V Velocity')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0, 1])

        # Pressure
        im3 = axes[1, 0].contourf(X, Y, self.p, levels=20, cmap='RdBu_r')
        axes[1, 0].set_title('Pressure')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 0])

        # Streamlines
        speed = np.sqrt(self.u ** 2 + self.v ** 2)
        axes[1, 1].streamplot(X, Y, self.u, self.v, color=speed, cmap='jet', density=1.5)
        axes[1, 1].set_title('Streamlines')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')

        # Add solid regions
        if np.any(self.solid_mask):
            for ax in axes.flat:
                solid_contour = ax.contour(X, Y, self.solid_mask.astype(float), levels=[0.5], colors='black', linewidths=2)

        plt.tight_layout()
        if save_prefix:
            plt.savefig(f'{save_prefix}_results.png', dpi=150)
        plt.show()

    def plot_centerline_velocity(self):
        j_center = self.ny // 2
        i_center = self.nx // 2

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # U velocity along vertical centerline
        ax1.plot(self.u[:, i_center], self.y, 'b-', linewidth=2, label='Computed')
        ax1.set_xlabel('U velocity')
        ax1.set_ylabel('y')
        ax1.set_title('U velocity along vertical centerline')
        ax1.grid(True)
        ax1.legend()

        # V velocity along horizontal centerline
        ax2.plot(self.x, self.v[j_center, :], 'r-', linewidth=2, label='Computed')
        ax2.set_xlabel('x')
        ax2.set_ylabel('V velocity')
        ax2.set_title('V velocity along horizontal centerline')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Problem 1: Lid-driven cavity at Re=100
    print("Problem 1: Lid-driven cavity flow")
    solver1 = Solver(nx=65, ny=65, Re=100, problem_type='cavity')
    solver1.solve(max_iter=1000, tolerance=1e-5)
    solver1.plot_results(save_prefix='cavity_Re100')
    solver1.plot_centerline_velocity()
