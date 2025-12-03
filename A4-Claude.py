import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.patches as patches


class NavierStokesSolver:
    """
    2D Incompressible Navier-Stokes Solver using SIMPLE Method
    """

    def __init__(self, nx, ny, Re, problem_type='cavity'):
        """
        Initialize the solver

        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x and y directions
        Re : float
            Reynolds number
        problem_type : str
            'cavity', 'cavity_step', or 'backstep'
        """
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.problem_type = problem_type

        # Setup geometry based on problem type
        self._setup_geometry()

        # Initialize fields
        self.u = np.zeros((ny, nx))  # x-velocity
        self.v = np.zeros((ny, nx))  # y-velocity
        self.p = np.zeros((ny, nx))  # pressure

        # Velocity corrections
        self.u_star = np.zeros((ny, nx))
        self.v_star = np.zeros((ny, nx))
        self.p_prime = np.zeros((ny, nx))

        # Relaxation factors
        self.alpha_u = 0.7
        self.alpha_v = 0.7
        self.alpha_p = 0.3

    def _setup_geometry(self):
        """Setup computational domain and boundaries"""
        if self.problem_type == 'cavity':
            self.Lx = 1.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx - 1)
            self.dy = self.Ly / (self.ny - 1)
            self.x = np.linspace(0, self.Lx, self.nx)
            self.y = np.linspace(0, self.Ly, self.ny)
            self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool)

        elif self.problem_type == 'cavity_step':
            self.Lx = 1.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx - 1)
            self.dy = self.Ly / (self.ny - 1)
            self.x = np.linspace(0, self.Lx, self.nx)
            self.y = np.linspace(0, self.Ly, self.ny)

            # Create solid step in bottom-left corner (1/4 of domain)
            self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool)
            step_x = self.nx // 4
            step_y = self.ny // 4
            self.solid_mask[:step_y, :step_x] = True

        elif self.problem_type == 'backstep':
            self.Lx = 2.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx - 1)
            self.dy = self.Ly / (self.ny - 1)
            self.x = np.linspace(0, self.Lx, self.nx)
            self.y = np.linspace(0, self.Ly, self.ny)

            # Create backstep (lower half of inlet blocked)
            self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool)
            step_y = self.ny // 2
            self.solid_mask[:step_y, 0] = True

    def _apply_boundary_conditions(self):
        """Apply boundary conditions based on problem type"""
        if self.problem_type == 'cavity':
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

            # Solid step boundaries
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

    def _solve_momentum_u(self):
        """Solve u-momentum equation with under-relaxation"""
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)

        nu = 1.0 / self.Re

        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if self.solid_mask[j, i]:
                    continue

                idx = j * self.nx + i

                # Diffusion coefficients
                De = nu * self.dy / self.dx
                Dw = nu * self.dy / self.dx
                Dn = nu * self.dx / self.dy
                Ds = nu * self.dx / self.dy

                # Convective fluxes (upwind)
                ue = 0.5 * (self.u[j, i] + self.u[j, i + 1])
                uw = 0.5 * (self.u[j, i] + self.u[j, i - 1])
                un = 0.5 * (self.u[j, i] + self.u[j + 1, i])
                us = 0.5 * (self.u[j, i] + self.u[j - 1, i])

                vn = 0.5 * (self.v[j + 1, i] + self.v[j + 1, i - 1])
                vs = 0.5 * (self.v[j, i] + self.v[j, i - 1])

                Fe = ue * self.dy
                Fw = uw * self.dy
                Fn = vn * self.dx
                Fs = vs * self.dx

                # Upwind scheme for convection
                aE = De + max(-Fe, 0)
                aW = Dw + max(Fw, 0)
                aN = Dn + max(-Fn, 0)
                aS = Ds + max(Fs, 0)

                # Under-relaxation
                aP0 = 0.0  # For steady state
                aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs) + aP0
                aP /= self.alpha_u

                # Pressure gradient
                dp_dx = (self.p[j, i + 1] - self.p[j, i - 1]) / (2 * self.dx)

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

                b[idx] = -dp_dx * self.dx * self.dy + (1 - self.alpha_u) * aP * self.u[j, i]

        # Apply boundary conditions to matrix
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1 or self.solid_mask[j, i]:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = self.u[j, i]

        A = A.tocsr()
        u_flat = spsolve(A, b)
        self.u_star = u_flat.reshape((self.ny, self.nx))

    def _solve_momentum_v(self):
        """Solve v-momentum equation with under-relaxation"""
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)

        nu = 1.0 / self.Re

        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if self.solid_mask[j, i]:
                    continue

                idx = j * self.nx + i

                # Diffusion coefficients
                De = nu * self.dy / self.dx
                Dw = nu * self.dy / self.dx
                Dn = nu * self.dx / self.dy
                Ds = nu * self.dx / self.dy

                # Convective fluxes
                ue = 0.5 * (self.u[j, i + 1] + self.u[j - 1, i + 1])
                uw = 0.5 * (self.u[j, i] + self.u[j - 1, i])

                ve = 0.5 * (self.v[j, i] + self.v[j, i + 1])
                vw = 0.5 * (self.v[j, i] + self.v[j, i - 1])
                vn = 0.5 * (self.v[j, i] + self.v[j + 1, i])
                vs = 0.5 * (self.v[j, i] + self.v[j - 1, i])

                Fe = ue * self.dy
                Fw = uw * self.dy
                Fn = vn * self.dx
                Fs = vs * self.dx

                # Upwind scheme
                aE = De + max(-Fe, 0)
                aW = Dw + max(Fw, 0)
                aN = Dn + max(-Fn, 0)
                aS = Ds + max(Fs, 0)

                aP0 = 0.0
                aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs) + aP0
                aP /= self.alpha_v

                # Pressure gradient
                dp_dy = (self.p[j + 1, i] - self.p[j - 1, i]) / (2 * self.dy)

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

                b[idx] = -dp_dy * self.dx * self.dy + (1 - self.alpha_v) * aP * self.v[j, i]

        # Apply boundary conditions
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1 or self.solid_mask[j, i]:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = self.v[j, i]

        A = A.tocsr()
        v_flat = spsolve(A, b)
        self.v_star = v_flat.reshape((self.ny, self.nx))

    def _solve_pressure_correction(self):
        """Solve pressure correction equation (Poisson equation)"""
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)

        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if self.solid_mask[j, i]:
                    continue

                idx = j * self.nx + i

                # Coefficients for pressure correction
                aE = self.dy / self.dx
                aW = self.dy / self.dx
                aN = self.dx / self.dy
                aS = self.dx / self.dy
                aP = aE + aW + aN + aS

                # Mass source (continuity violation)
                div_u = ((self.u_star[j, i + 1] - self.u_star[j, i - 1]) / (2 * self.dx) +
                         (self.v_star[j + 1, i] - self.v_star[j - 1, i]) / (2 * self.dy))

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

                b[idx] = div_u * self.dx * self.dy

        # Pressure boundary conditions (Neumann)
        for i in range(self.nx):
            # Bottom
            idx = 0 * self.nx + i
            A[idx, :] = 0
            A[idx, idx] = 1
            if i > 0 and i < self.nx - 1:
                A[idx, idx + self.nx] = -1
            b[idx] = 0

            # Top
            idx = (self.ny - 1) * self.nx + i
            A[idx, :] = 0
            A[idx, idx] = 1
            if i > 0 and i < self.nx - 1:
                A[idx, idx - self.nx] = -1
            b[idx] = 0

        for j in range(1, self.ny - 1):
            # Left
            idx = j * self.nx + 0
            A[idx, :] = 0
            A[idx, idx] = 1
            A[idx, idx + 1] = -1
            b[idx] = 0

            # Right
            idx = j * self.nx + (self.nx - 1)
            A[idx, :] = 0
            A[idx, idx] = 1
            A[idx, idx - 1] = -1
            b[idx] = 0

        # Reference pressure (set one point to zero)
        ref_idx = (self.ny // 2) * self.nx + (self.nx // 2)
        A[ref_idx, :] = 0
        A[ref_idx, ref_idx] = 1
        b[ref_idx] = 0

        A = A.tocsr()
        p_prime_flat = spsolve(A, b)
        self.p_prime = p_prime_flat.reshape((self.ny, self.nx))

    def _correct_velocity_pressure(self):
        """Correct velocities and pressure"""
        # Update pressure
        self.p += self.alpha_p * self.p_prime

        # Update velocities
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if not self.solid_mask[j, i]:
                    dp_dx = (self.p_prime[j, i + 1] - self.p_prime[j, i - 1]) / (2 * self.dx)
                    dp_dy = (self.p_prime[j + 1, i] - self.p_prime[j - 1, i]) / (2 * self.dy)

                    self.u[j, i] = self.u_star[j, i] - self.dx * dp_dx
                    self.v[j, i] = self.v_star[j, i] - self.dy * dp_dy

        self._apply_boundary_conditions()

    def solve(self, max_iter=1000, tolerance=1e-6):
        """
        Solve the Navier-Stokes equations using SIMPLE algorithm
        """
        print(f"Solving {self.problem_type} at Re={self.Re}")

        for iteration in range(max_iter):
            u_old = self.u.copy()
            v_old = self.v.copy()

            # SIMPLE algorithm steps
            self._solve_momentum_u()
            self._solve_momentum_v()
            self._solve_pressure_correction()
            self._correct_velocity_pressure()

            # Check convergence
            u_diff = np.max(np.abs(self.u - u_old))
            v_diff = np.max(np.abs(self.v - v_old))
            max_diff = max(u_diff, v_diff)

            if iteration % 10 == 0:
                div_max = self._compute_divergence()
                print(f"Iteration {iteration}: max_diff={max_diff:.6e}, max_div={div_max:.6e}")

            if max_diff < tolerance:
                print(f"Converged in {iteration} iterations")
                break

        return self.u, self.v, self.p

    def _compute_divergence(self):
        """Compute maximum divergence of velocity field"""
        div = np.zeros((self.ny, self.nx))
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if not self.solid_mask[j, i]:
                    div[j, i] = ((self.u[j, i + 1] - self.u[j, i - 1]) / (2 * self.dx) +
                                 (self.v[j + 1, i] - self.v[j - 1, i]) / (2 * self.dy))
        return np.max(np.abs(div))

    def plot_results(self, save_prefix=''):
        """Plot velocity, pressure, and streamlines"""
        X, Y = np.meshgrid(self.x, self.y)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # U velocity
        im1 = axes[0, 0].contourf(X, Y, self.u, levels=20, cmap='RdBu_r')
        axes[0, 0].set_title('U Velocity')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 0])

        # V velocity
        im2 = axes[0, 1].contourf(X, Y, self.v, levels=20, cmap='RdBu_r')
        axes[0, 1].set_title('V Velocity')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0, 1])

        # Pressure
        im3 = axes[1, 0].contourf(X, Y, self.p, levels=20, cmap='viridis')
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
                solid_contour = ax.contour(X, Y, self.solid_mask.astype(float),
                                           levels=[0.5], colors='black', linewidths=2)

        plt.tight_layout()
        if save_prefix:
            plt.savefig(f'{save_prefix}_results.png', dpi=150)
        plt.show()

    def plot_centerline_velocity(self):
        """Plot velocity profile through cavity center"""
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


# Example usage
if __name__ == "__main__":
    # Problem 1: Lid-driven cavity at Re=100
    print("=" * 60)
    print("Problem 1: Lid-driven cavity flow")
    print("=" * 60)
    solver1 = NavierStokesSolver(nx=65, ny=65, Re=100, problem_type='cavity')
    solver1.solve(max_iter=500, tolerance=1e-5)
    solver1.plot_results(save_prefix='cavity_Re100')
    solver1.plot_centerline_velocity()

    # Problem 2: Lid-driven cavity with step at Re=200
    print("\n" + "=" * 60)
    print("Problem 2: Lid-driven cavity with step")
    print("=" * 60)
    solver2 = NavierStokesSolver(nx=80, ny=80, Re=200, problem_type='cavity_step')
    solver2.solve(max_iter=800, tolerance=1e-5)
    solver2.plot_results(save_prefix='cavity_step_Re200')

    # Problem 3 (BONUS): Back-step flow
    print("\n" + "=" * 60)
    print("Problem 3: Back-step flow")
    print("=" * 60)
    for Re in [100, 200, 400]:
        print(f"\nSolving for Re={Re}")
        solver3 = NavierStokesSolver(nx=160, ny=80, Re=Re, problem_type='backstep')
        solver3.solve(max_iter=1000, tolerance=1e-5)
        solver3.plot_results(save_prefix=f'backstep_Re{Re}')