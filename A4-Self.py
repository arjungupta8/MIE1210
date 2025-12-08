import numpy as np
import matplotlib.pyplot as plt


class solver:
    def __init__(self, nx, ny, Re, problem_type):
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.problem_type = problem_type
        self.nu = 1.0/ self.Re

        self.u = np.zeros((ny, nx)) # Velocity and pressure fields
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))

        self.u_star = np.zeros_like(self.u) # Correction fields
        self.v_star = np.zeros_like(self.v)
        self.p_prime = np.zeros_like(self.p)

        self.d_u = np.ones((ny, nx)) * 1e-10 # D coefficients
        self.d_v = np.ones((ny, nx)) * 1e-10

        self.alpha_u = 0.7 # Under relaxation variables
        self.alpha_v = 0.7
        self.alpha_p = 0.3

        # Convergence criteria
        self.tolerance = 1e-5
        self.max_iterations = 1000

        self.solid_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self._setup_geometry()


    def _setup_geometry(self):
        if self.problem_type == 'cavity':
            self.Lx = 1.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx - 1)
            self.dy = self.Ly / (self.ny - 1)
            self.x = np.linspace(0, self.Lx, self.nx)
            self.y = np.linspace(0, self.Ly, self.ny)


        elif self.problem_type == 'cavity_step':
            self.Lx = 1.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx - 1)
            self.dy = self.Ly / (self.ny - 1)
            self.x = np.linspace(0, self.Lx, self.nx)
            self.y = np.linspace(0, self.Ly, self.ny)
            step_x = self.nx // 3
            step_y = self.ny // 3
            self.solid_mask[:step_y, :step_x] = True

        elif self.problem_type == 'backstep':
            self.Lx = 2.0
            self.Ly = 0.25
            self.dx = self.Lx / (self.nx - 1)
            self.dy = self.Lx / (self.ny - 1)
            self.x = np.linspace(0, self.Lx, self.nx)
            self.y = np.linspace(0, self.Ly, self.ny)
            step_x = self.nx // 2
            step_y = self.ny // 2
            self.solid_mask[:step_y, :step_x] = True

    def apply_bc(self):
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
            self.u[-1,:] = 1.0

        if self.problem_type == 'backstep': # NO PARABOLIC PROFILE
            j_mid = self.ny // 2
            self.u[j_mid:,0] = 1.0
            self.u[:, -1] = self.u[:,-2] # Zero Gradient at outlet (approx)
            self.v[:,-1] = self.v[:,-2]

        self.u[self.solid_mask] = 0.0
        self.v[self.solid_mask] = 0.0

        # 2nd: Pressure BCs

        # Zero gradient at Boundaries (approximate Neumann)
        self.p[0, :] = self.p[1, :]
        self.p[-1, :] = self.p[-2, :]
        self.p[:, 0] = self.p[:, 1]
        self.p[:, -1] = self.p[:, -2]
        # Reference Pressure at domain center (p_ref = 0)
        j_ref = self.ny // 2
        i_ref = self.nx // 2
        self.p[j_ref, i_ref] = 0.0





if __name__ == "__main__":

    cavitySolve = solver(nx = 33, ny = 33, Re = 100, problem_type = 'cavity')
    cavityStepSolve = solver(nx=33, ny=33, Re=100, problem_type='cavity_step')
    # backstepSolve = solver(nx = 33, ny=33, Re=100, problem_type='backstep')

    # Actual Assignment Cases:
    # cavitySolve = solver(257, 257, 100, 'cavity')
    # cavityStepSolve = solver(320, 320, 200, 'cavity_step'
    # backstepSolve = solver(320, 160, 100, 'backstep')
    # backstepSolve = solver(320, 160, 200, 'backstep')
    # backstepSolve = solver(320, 160, 400, 'backstep')



