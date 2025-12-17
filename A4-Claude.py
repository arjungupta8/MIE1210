import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg

# THIS IS FRACTIONAL-STEP, NOT SIMPLE

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

        # For Rhie-Chow interpolation
        self.d_u = np.ones((ny, nx)) * 1e-10  # Momentum coefficient storage
        self.d_v = np.ones((ny, nx)) * 1e-10

        # Conservative relaxation
        self.alpha_u = 0.7
        self.alpha_v = 0.7
        self.alpha_p = 0.2

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

        # Zero d coefficients at boundaries
        self.d_u[0, :] = 0.0
        self.d_u[-1, :] = 0.0
        self.d_u[:, 0] = 0.0
        self.d_u[:, -1] = 0.0
        self.d_v[0, :] = 0.0
        self.d_v[-1, :] = 0.0
        self.d_v[:, 0] = 0.0
        self.d_v[:, -1] = 0.0

    def _rhie_chow_u_face(self, j, i, face):
        """
        Rhie-Chow interpolation for u-velocity at cell faces
        Prevents pressure-velocity decoupling
        """
        if face == 'east':
            if i >= self.nx - 1:
                return self.u[j, i]

            # Average velocity
            u_avg = 0.5 * (self.u[j, i] + self.u[j, i + 1])

            # Average d coefficient
            d_avg = 0.5 * (self.d_u[j, i] + self.d_u[j, i + 1])

            # Pressure gradient at face (direct)
            dp_dx_face = (self.p[j, i + 1] - self.p[j, i]) / self.dx

            # Pressure gradients at cell centers (central difference)
            dp_dx_P = (self.p[j, i + 1] - self.p[j, i - 1]) / (2 * self.dx) if i > 0 else 0
            dp_dx_E = (self.p[j, i + 2] - self.p[j, i]) / (2 * self.dx) if i < self.nx - 2 else 0
            dp_dx_avg = 0.5 * (dp_dx_P + dp_dx_E)

            # Rhie-Chow correction
            u_face = u_avg - d_avg * (dp_dx_face - dp_dx_avg)
            return u_face

        elif face == 'west':
            if i <= 0:
                return self.u[j, i]

            u_avg = 0.5 * (self.u[j, i - 1] + self.u[j, i])
            d_avg = 0.5 * (self.d_u[j, i - 1] + self.d_u[j, i])

            dp_dx_face = (self.p[j, i] - self.p[j, i - 1]) / self.dx

            dp_dx_W = (self.p[j, i] - self.p[j, i - 2]) / (2 * self.dx) if i > 1 else 0
            dp_dx_P = (self.p[j, i + 1] - self.p[j, i - 1]) / (2 * self.dx) if i < self.nx - 1 else 0
            dp_dx_avg = 0.5 * (dp_dx_W + dp_dx_P)

            u_face = u_avg - d_avg * (dp_dx_face - dp_dx_avg)
            return u_face

        elif face == 'north':
            if j >= self.ny - 1:
                return self.u[j, i]

            u_avg = 0.5 * (self.u[j, i] + self.u[j + 1, i])
            d_avg = 0.5 * (self.d_u[j, i] + self.d_u[j + 1, i])

            dp_dy_face = (self.p[j + 1, i] - self.p[j, i]) / self.dy

            dp_dy_P = (self.p[j + 1, i] - self.p[j - 1, i]) / (2 * self.dy) if j > 0 else 0
            dp_dy_N = (self.p[j + 2, i] - self.p[j, i]) / (2 * self.dy) if j < self.ny - 2 else 0
            dp_dy_avg = 0.5 * (dp_dy_P + dp_dy_N)

            u_face = u_avg - d_avg * (dp_dy_face - dp_dy_avg)
            return u_face

        elif face == 'south':
            if j <= 0:
                return self.u[j, i]

            u_avg = 0.5 * (self.u[j - 1, i] + self.u[j, i])
            d_avg = 0.5 * (self.d_u[j - 1, i] + self.d_u[j, i])

            dp_dy_face = (self.p[j, i] - self.p[j - 1, i]) / self.dy

            dp_dy_S = (self.p[j, i] - self.p[j - 2, i]) / (2 * self.dy) if j > 1 else 0
            dp_dy_P = (self.p[j + 1, i] - self.p[j - 1, i]) / (2 * self.dy) if j < self.ny - 1 else 0
            dp_dy_avg = 0.5 * (dp_dy_S + dp_dy_P)

            u_face = u_avg - d_avg * (dp_dy_face - dp_dy_avg)
            return u_face

    def _rhie_chow_v_face(self, j, i, face):
        """
        Rhie-Chow interpolation for v-velocity at cell faces
        """
        if face == 'north':
            if j >= self.ny - 1:
                return self.v[j, i]

            v_avg = 0.5 * (self.v[j, i] + self.v[j + 1, i])
            d_avg = 0.5 * (self.d_v[j, i] + self.d_v[j + 1, i])

            dp_dy_face = (self.p[j + 1, i] - self.p[j, i]) / self.dy

            dp_dy_P = (self.p[j + 1, i] - self.p[j - 1, i]) / (2 * self.dy) if j > 0 else 0
            dp_dy_N = (self.p[j + 2, i] - self.p[j, i]) / (2 * self.dy) if j < self.ny - 2 else 0
            dp_dy_avg = 0.5 * (dp_dy_P + dp_dy_N)

            v_face = v_avg - d_avg * (dp_dy_face - dp_dy_avg)
            return v_face

        elif face == 'south':
            if j <= 0:
                return self.v[j, i]

            v_avg = 0.5 * (self.v[j - 1, i] + self.v[j, i])
            d_avg = 0.5 * (self.d_v[j - 1, i] + self.d_v[j, i])

            dp_dy_face = (self.p[j, i] - self.p[j - 1, i]) / self.dy

            dp_dy_S = (self.p[j, i] - self.p[j - 2, i]) / (2 * self.dy) if j > 1 else 0
            dp_dy_P = (self.p[j + 1, i] - self.p[j - 1, i]) / (2 * self.dy) if j < self.ny - 1 else 0
            dp_dy_avg = 0.5 * (dp_dy_S + dp_dy_P)

            v_face = v_avg - d_avg * (dp_dy_face - dp_dy_avg)
            return v_face

        elif face == 'east':
            if i >= self.nx - 1:
                return self.v[j, i]

            v_avg = 0.5 * (self.v[j, i] + self.v[j, i + 1])
            d_avg = 0.5 * (self.d_v[j, i] + self.d_v[j, i + 1])

            dp_dx_face = (self.p[j, i + 1] - self.p[j, i]) / self.dx

            dp_dx_P = (self.p[j, i + 1] - self.p[j, i - 1]) / (2 * self.dx) if i > 0 else 0
            dp_dx_E = (self.p[j, i + 2] - self.p[j, i]) / (2 * self.dx) if i < self.nx - 2 else 0
            dp_dx_avg = 0.5 * (dp_dx_P + dp_dx_E)

            v_face = v_avg - d_avg * (dp_dx_face - dp_dx_avg)
            return v_face

        elif face == 'west':
            if i <= 0:
                return self.v[j, i]

            v_avg = 0.5 * (self.v[j, i - 1] + self.v[j, i])
            d_avg = 0.5 * (self.d_v[j, i - 1] + self.d_v[j, i])

            dp_dx_face = (self.p[j, i] - self.p[j, i - 1]) / self.dx

            dp_dx_W = (self.p[j, i] - self.p[j, i - 2]) / (2 * self.dx) if i > 1 else 0
            dp_dx_P = (self.p[j, i + 1] - self.p[j, i - 1]) / (2 * self.dx) if i < self.nx - 1 else 0
            dp_dx_avg = 0.5 * (dp_dx_W + dp_dx_P)

            v_face = v_avg - d_avg * (dp_dx_face - dp_dx_avg)
            return v_face

    def solve(self, dt=0.001, max_steps=100000, check_interval=100, tolerance=1e-6):
        """
        Fractional step method with Rhie-Chow interpolation
        """
        print(f"Solving {self.problem_type} at Re={self.Re}")
        print(f"Grid: {self.nx}x{self.ny}")
        print(f"Time step: dt={dt}")
        print(f"Using Rhie-Chow interpolation for momentum")

        self._apply_boundary_conditions()

        nu = 1.0 / self.Re

        for step in range(max_steps):
            u_old = self.u.copy()
            v_old = self.v.copy()

            # Step 1: Solve momentum with Rhie-Chow
            u_star = self._solve_u_momentum_rhie_chow(dt, nu)
            v_star = self._solve_v_momentum_rhie_chow(dt, nu)

            # Step 2: Pressure correction
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

    def _solve_u_momentum_rhie_chow(self, dt, nu):
        """Solve u-momentum using Rhie-Chow interpolated face velocities"""
        u_star = self.u.copy()

        # First compute d_u coefficients for all interior cells
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                # Compute momentum equation coefficient (simplified)
                diff_coeff = nu * (2.0 / self.dx ** 2 + 2.0 / self.dy ** 2)
                aP = 1.0 / dt + diff_coeff
                self.d_u[j, i] = 1.0 / max(aP, 1e-10)

        # Now solve momentum with Rhie-Chow face velocities
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                # Use Rhie-Chow for face velocities in convection
                u_e = self._rhie_chow_u_face(j, i, 'east')
                u_w = self._rhie_chow_u_face(j, i, 'west')
                u_n = self._rhie_chow_u_face(j, i, 'north')
                u_s = self._rhie_chow_u_face(j, i, 'south')

                v_n = self._rhie_chow_v_face(j, i, 'north')
                v_s = self._rhie_chow_v_face(j, i, 'south')

                # Convection term with Rhie-Chow velocities
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

    def _solve_v_momentum_rhie_chow(self, dt, nu):
        """Solve v-momentum using Rhie-Chow interpolated face velocities"""
        v_star = self.v.copy()

        # First compute d_v coefficients
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                diff_coeff = nu * (2.0 / self.dx ** 2 + 2.0 / self.dy ** 2)
                aP = 1.0 / dt + diff_coeff
                self.d_v[j, i] = 1.0 / max(aP, 1e-10)

        # Solve momentum with Rhie-Chow
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                # Use Rhie-Chow for face velocities
                v_e = self._rhie_chow_v_face(j, i, 'east')
                v_w = self._rhie_chow_v_face(j, i, 'west')
                v_n = self._rhie_chow_v_face(j, i, 'north')
                v_s = self._rhie_chow_v_face(j, i, 'south')

                u_e = self._rhie_chow_u_face(j, i, 'east')
                u_w = self._rhie_chow_u_face(j, i, 'west')

                # Convection with Rhie-Chow velocities
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

                # Boundaries: Neumann BC
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

        # Reference pressure
        ref_idx = self.ny // 2 * self.nx + self.nx // 2
        A[ref_idx, :] = 0
        A[ref_idx, ref_idx] = 1.0
        b[ref_idx] = 0.0

        A = A.tocsr()
        p_flat, info = cg(A, b)
        p_new = p_flat.reshape((self.ny, self.nx))

        # Update with relaxation
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

    def plot_results(self):
        X, Y = np.meshgrid(self.x, self.y)
        nx, ny = self.nx, self.ny

        # ---------------- U velocity ----------------
        plt.figure(figsize=(6, 5))
        im = plt.contourf(X, Y, self.u, levels=20, cmap='RdBu_r')
        plt.colorbar(im)
        plt.title('U Velocity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'{nx}x{ny}-u_velocity-a4_claude.png', dpi=150)
        plt.show()

        # ---------------- V velocity ----------------
        plt.figure(figsize=(6, 5))
        im = plt.contourf(X, Y, self.v, levels=20, cmap='RdBu_r')
        plt.colorbar(im)
        plt.title('V Velocity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'{nx}x{ny}-v_velocity-a4_claude.png', dpi=150)
        plt.show()

        # ---------------- Pressure ----------------
        plt.figure(figsize=(6, 5))
        im = plt.contourf(X, Y, self.p, levels=20, cmap='RdBu_r')
        plt.colorbar(im)
        plt.title('Pressure')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'{nx}x{ny}-pressure-a4_claude.png', dpi=150)
        plt.show()

        # ---------------- Streamlines ----------------
        speed = np.sqrt(self.u ** 2 + self.v ** 2)
        plt.figure(figsize=(6, 5))
        plt.streamplot(X, Y, self.u, self.v, color=speed, cmap='jet', density=1.5)
        plt.title('Streamlines')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'{nx}x{ny}-streamlines-a4_claude.png', dpi=150)
        plt.show()

    def plot_centerline_velocity(self):
        nx, ny = self.nx, self.ny
        j_center = ny // 2
        i_center = nx // 2

        # -------- Vertical centerline (U) --------
        plt.figure(figsize=(6, 5))
        plt.plot(self.u[:, i_center], self.y, 'b-', linewidth=2)
        plt.xlabel('U velocity')
        plt.ylabel('y')
        plt.title('U velocity along vertical centerline')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{nx}x{ny}-u_centerline-a4_claude.png', dpi=150)
        plt.show()

        # -------- Horizontal centerline (V) --------
        plt.figure(figsize=(6, 5))
        plt.plot(self.x, self.v[j_center, :], 'r-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('V velocity')
        plt.title('V velocity along horizontal centerline')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{nx}x{ny}-v_centerline-a4_claude.png', dpi=150)
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Lid-driven cavity with Rhie-Chow Interpolation")
    print("=" * 60)

    solver = Solver(nx=257, ny=257, Re=100, problem_type='cavity')
    solver.solve(dt=0.001, max_steps=50000, check_interval=100, tolerance=5e-5)

    solver.plot_results(save_prefix='cavity_Re100_rhie_chow')
    solver.plot_centerline_velocity()