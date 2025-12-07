import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# THIS CODE WORKS FOR THE FIRST PROBLEM


class Solver:
    def __init__(self, nx: int, ny: int, Re: float, problem_type: str):
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.problem_type = problem_type

        self._setup_geometry()

        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))

        # Conservative relaxation
        self.alpha_u = 0.7
        self.alpha_v = 0.7
        self.alpha_p = 0.3

    def _setup_geometry(self):
        if self.problem_type == 'cavity':
            self.Lx = 1.0
            self.Ly = 1.0
            self.dx = self.Lx / (self.nx - 1)
            self.dy = self.Ly / (self.ny - 1)
            self.x = np.linspace(0, self.Lx, self.nx)
            self.y = np.linspace(0, self.Ly, self.ny)

    def _apply_boundary_conditions(self):
        # Top wall moves with u=1 (lid)
        self.u[-1, :] = 1.0
        self.v[-1, :] = 0.0

        # Bottom, left, right walls: no-slip
        self.u[0, :] = 0.0
        self.v[0, :] = 0.0
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[:, -1] = 0.0

    def solve(self, dt=0.001, max_steps=100000, check_interval=100, tolerance=1e-6):
        """
        Fractional step method for incompressible Navier-Stokes
        Much more stable than SIMPLE for beginners
        """
        print(f"Solving {self.problem_type} at Re={self.Re}")
        print(f"Grid: {self.nx}x{self.ny}")
        print(f"Time step: dt={dt}")

        self._apply_boundary_conditions()

        nu = 1.0 / self.Re

        for step in range(max_steps):
            u_old = self.u.copy()
            v_old = self.v.copy()

            # Step 1: Advection-diffusion (predict velocities)
            u_star = self._solve_u_momentum(dt, nu)
            v_star = self._solve_v_momentum(dt, nu)

            # Step 2: Pressure correction (enforce continuity)
            self._solve_pressure_poisson(u_star, v_star, dt)

            # Step 3: Correct velocities
            self._correct_velocities(u_star, v_star, dt)

            # Apply BCs
            self._apply_boundary_conditions()

            # Check convergence
            if step % check_interval == 0:
                du = np.max(np.abs(self.u - u_old))
                dv = np.max(np.abs(self.v - v_old))
                div_max = self._compute_divergence()
                print(f"Step {step:5d}: du={du:.6e}, dv={dv:.6e}, div={div_max:.6e}")

                if du < tolerance and dv < tolerance:
                    print(f"\nConverged at step {step}")
                    print(f"Final divergence: {div_max:.6e}")
                    return self.u, self.v, self.p

        print(f"\nReached max steps")
        return self.u, self.v, self.p

    def _solve_u_momentum(self, dt, nu):
        """Solve u-momentum with implicit diffusion, explicit convection"""
        u_star = self.u.copy()

        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                # Convection (explicit)
                u_e = 0.5 * (self.u[j, i] + self.u[j, i + 1])
                u_w = 0.5 * (self.u[j, i - 1] + self.u[j, i])
                u_n = 0.5 * (self.u[j, i] + self.u[j + 1, i])
                u_s = 0.5 * (self.u[j - 1, i] + self.u[j, i])

                v_n = 0.5 * (self.v[j, i] + self.v[j + 1, i])
                v_s = 0.5 * (self.v[j - 1, i] + self.v[j, i])

                conv = (u_e ** 2 - u_w ** 2) / self.dx + (u_n * v_n - u_s * v_s) / self.dy

                # Diffusion (explicit)
                diff = nu * ((self.u[j, i + 1] - 2 * self.u[j, i] + self.u[j, i - 1]) / self.dx ** 2 +
                             (self.u[j + 1, i] - 2 * self.u[j, i] + self.u[j - 1, i]) / self.dy ** 2)

                # Pressure gradient
                dp_dx = (self.p[j, i + 1] - self.p[j, i - 1]) / (2 * self.dx)

                # Update with under-relaxation
                u_star[j, i] = self.u[j, i] + dt * (-conv + diff - dp_dx)
                u_star[j, i] = self.alpha_u * u_star[j, i] + (1 - self.alpha_u) * self.u[j, i]

        return u_star

    def _solve_v_momentum(self, dt, nu):
        """Solve v-momentum with implicit diffusion, explicit convection"""
        v_star = self.v.copy()

        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                # Convection (explicit)
                v_e = 0.5 * (self.v[j, i] + self.v[j, i + 1])
                v_w = 0.5 * (self.v[j, i - 1] + self.v[j, i])
                v_n = 0.5 * (self.v[j, i] + self.v[j + 1, i])
                v_s = 0.5 * (self.v[j - 1, i] + self.v[j, i])

                u_e = 0.5 * (self.u[j, i] + self.u[j, i + 1])
                u_w = 0.5 * (self.u[j, i - 1] + self.u[j, i])

                conv = (u_e * v_e - u_w * v_w) / self.dx + (v_n ** 2 - v_s ** 2) / self.dy

                # Diffusion (explicit)
                diff = nu * ((self.v[j, i + 1] - 2 * self.v[j, i] + self.v[j, i - 1]) / self.dx ** 2 +
                             (self.v[j + 1, i] - 2 * self.v[j, i] + self.v[j - 1, i]) / self.dy ** 2)

                # Pressure gradient
                dp_dy = (self.p[j + 1, i] - self.p[j - 1, i]) / (2 * self.dy)

                # Update with under-relaxation
                v_star[j, i] = self.v[j, i] + dt * (-conv + diff - dp_dy)
                v_star[j, i] = self.alpha_v * v_star[j, i] + (1 - self.alpha_v) * self.v[j, i]

        return v_star

    def _solve_pressure_poisson(self, u_star, v_star, dt):
        """Solve pressure Poisson equation"""
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i

                # Boundaries: Neumann BC (dp/dn = 0)
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                    A[idx, idx] = 1.0
                    if i == 0 and i + 1 < self.nx:
                        A[idx, idx + 1] = -1.0
                    elif i == self.nx - 1 and i - 1 >= 0:
                        A[idx, idx - 1] = -1.0
                    elif j == 0 and j + 1 < self.ny:
                        A[idx, idx + self.nx] = -1.0
                    elif j == self.ny - 1 and j - 1 >= 0:
                        A[idx, idx - self.nx] = -1.0
                    b[idx] = 0.0
                else:
                    # Interior: Laplacian(p) = div(u*) / dt
                    A[idx, idx] = -2.0 / self.dx ** 2 - 2.0 / self.dy ** 2
                    A[idx, idx + 1] = 1.0 / self.dx ** 2
                    A[idx, idx - 1] = 1.0 / self.dx ** 2
                    A[idx, idx + self.nx] = 1.0 / self.dy ** 2
                    A[idx, idx - self.nx] = 1.0 / self.dy ** 2

                    # RHS: divergence of predicted velocity
                    div = (u_star[j, i + 1] - u_star[j, i - 1]) / (2 * self.dx) + \
                          (v_star[j + 1, i] - v_star[j - 1, i]) / (2 * self.dy)
                    b[idx] = div / dt

        # Reference pressure at center
        ref_idx = self.ny // 2 * self.nx + self.nx // 2
        A[ref_idx, :] = 0
        A[ref_idx, ref_idx] = 1.0
        b[ref_idx] = 0.0

        A = A.tocsr()
        p_flat = spsolve(A, b)
        p_new = p_flat.reshape((self.ny, self.nx))

        # Update pressure with relaxation
        self.p = self.alpha_p * p_new + (1 - self.alpha_p) * self.p

    def _correct_velocities(self, u_star, v_star, dt):
        """Correct velocities to satisfy continuity"""
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                self.u[j, i] = u_star[j, i] - dt * (self.p[j, i + 1] - self.p[j, i - 1]) / (2 * self.dx)
                self.v[j, i] = v_star[j, i] - dt * (self.p[j + 1, i] - self.p[j - 1, i]) / (2 * self.dy)

    def _compute_divergence(self):
        div_max = 0.0
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                div = abs((self.u[j, i + 1] - self.u[j, i - 1]) / (2 * self.dx) +
                          (self.v[j + 1, i] - self.v[j - 1, i]) / (2 * self.dy))
                div_max = max(div_max, div)
        return div_max

    def plot_results(self, save_prefix=''):
        X, Y = np.meshgrid(self.x, self.y)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        im1 = axes[0, 0].contourf(X, Y, self.u, levels=20, cmap='RdBu_r')
        axes[0, 0].set_title('U Velocity')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].contourf(X, Y, self.v, levels=20, cmap='RdBu_r')
        axes[0, 1].set_title('V Velocity')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 1])

        im3 = axes[1, 0].contourf(X, Y, self.p, levels=20, cmap='RdBu_r')
        axes[1, 0].set_title('Pressure')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axes[1, 0])

        speed = np.sqrt(self.u ** 2 + self.v ** 2)
        axes[1, 1].streamplot(X, Y, self.u, self.v, color=speed, cmap='jet', density=1.5)
        axes[1, 1].set_title('Streamlines')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_aspect('equal')

        plt.tight_layout()
        if save_prefix:
            plt.savefig(f'{save_prefix}_results.png', dpi=150)
        plt.show()

    def plot_centerline_velocity(self):
        j_center = self.ny // 2
        i_center = self.nx // 2

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(self.u[:, i_center], self.y, 'b-', linewidth=2, label='Computed')
        ax1.set_xlabel('U velocity')
        ax1.set_ylabel('y')
        ax1.set_title('U velocity along vertical centerline')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(self.x, self.v[j_center, :], 'r-', linewidth=2, label='Computed')
        ax2.set_xlabel('x')
        ax2.set_ylabel('V velocity')
        ax2.set_title('V velocity along horizontal centerline')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Lid-driven cavity flow - Fractional Step Method")
    print("=" * 60)

    # Start with moderate resolution
    solver = Solver(nx=65, ny=65, Re=100, problem_type='cavity')
    solver.solve(dt=0.001, max_steps=50000, check_interval=50, tolerance=1e-6)

    solver.plot_results(save_prefix='cavity_Re100')
    solver.plot_centerline_velocity()