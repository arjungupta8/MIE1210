import numpy as np
from fontTools.varLib.models import allEqual
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class solver:
    def __init__(self, nx, ny, Re, problem_type):
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

        self.d_u = np.zeros((ny, nx)) # Rhie-Chow coefficient for x vel
        self.d_v = np.zeros((ny, nx)) # Rhie-Chow coefficient for y vel

        self.alpha_u = 0.7 # Relaxation in x vel
        self.alpha_v = 0.7 # Relaxation in y vel
        self.alpha_p = 0.3 # Relaxation in pressure


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
            self.solid_mask[:step_x, :step_y] = True # as bottom left corner near 0,0 has the step

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
            d_avg = 0.5 * self.d_u[j, i] + self.d_u[j, i+1] # average of d coefficients
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

        nu = 1.0/self.Re # kinematic viscosity

        for j in range (1, self.ny - 1):
            for i in range(self.nx - 1):
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
                self.d_u[j, i] = self.dx * self.dy / aP if aP != 0 else 0.0 # Store d_u coefficients for R-C before under-relaxation
                aP /= self.alpha_u # apply under-relaxation
                dp_dx = (self.p[j, i+1] - self.p[j, i-1]) / (2* self.dx) # pressure gradient

                # Build matrix
                A[idx, idx] = aP
                if i < self.nx-1:
                    A[idx, idx+1] = -aE
                if i > 0:
                    A[idx, idx-1] = -aW
                if j< self.ny - 1:
                    A[idx, idx+self.nx] = -aN
                if j > 0:
                    A[idx, idx-self.nx] = -aS

                b[idx] = -dp_dx * self.dx * self.dy + (1 - self.alpha_u) * aP * self.u[j,i] # Equation 14

        for j in range(self.ny): # Apply BCs to matrix
            for i in range(self.nx):
                idx = j * self.nx + i
                if i == 0 or i == self.nx-1 or j == 0 or j == self.ny-1 or self.solid_mask[j, i]:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = self.u[j,i]
                    self.d_u[j, i] = 0.0 # No correction at boundaries
        A = A.tocsr()
        u_flat = spsolve(A,b) # can change this to an iterative solver if required
        self.u_star = u_flat.reshape((self.ny, self.nx))


    def _solve_momentum_v(self):
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)
        nu = 1.0/self.Re

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
                self.d_v[j, i] = self.dx * self.dy / aP if aP != 0 else 0.0  # Store d_v coefficients for R-C before under-relaxation
                aP /= self.alpha_v  # apply under-relaxation
                dp_dy = (self.p[j+1, i] - self.p[j-1, i]) / (2 * self.dy)  # pressure gradient

                # Build matrix
                A[idx, idx] = aP
                if i < self.nx - 1:
                    A[idx, idx + 1] = -aE
                if i > 0:
                    A[idx, idx - 1] = -aW
                if j < self.ny - 1:
                    A[idx, idx + self.nx] = -aN
                if j > 0:
                    A[idx, idx - self.nx] = -aS

                b[idx] = -dp_dy * self.dx * self.dy + (1 - self.alpha_v) * aP * self.v[j, i] # Equation 14

        for j in range(self.ny): # Apply BCs to matrix
            for i in range(self.nx):
                idx = j * self.nx + i
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1 or self.solid_mask[j, i]:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = self.v[j, i]
                    self.d_v[j, i] = 0.0  # No correction at boundaries

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
                idx = j * self.nx + 1
                # Pressure Correction Coefficients, Equation 25, 26
                aE = self.dy / self.dx
                aW = self.dy / self.dx
                aN = self.dx / self.dy
                aS = self.dx / self.dy
                aP = aE + aW + aN + aS

                # Mass Source (Continuity violation). Equation 27, bracketed section
                div_u = ((self.u_star[j, i + 1] - self.u_star[j, i - 1]) / (2 * self.dx) +
                         (self.v_star[j + 1, i] - self.v_star[j - 1, i]) / (2 * self.dy))

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
                b[idx] = div_u * self.dx * self.dy





    def solve(self, max_iter = 1000, tolerance = 1e-6):
        print (f"Solving {self.problem_type} at Re={self.Re}")

        for iteration in range (max_iter):
            u_old = self.u.copy()
            v_old = self.v.copy()

            # SIMPLE steps
            self._solve_momentum_u()
            self._solve_momentum_v()
            self._solve_pressure_correction()


