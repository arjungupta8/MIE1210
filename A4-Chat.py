import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ====================================================
# SIMPLE-BASED NAVIER–STOKES SOLVER (Steady, 2D)
# ====================================================

class SimpleSolver:
    def __init__(self, nx, ny, Re, geometry="cavity"):
        self.nx, self.ny = nx, ny
        self.Re = Re
        self.nu = 1.0 / Re    # non-dimensional kinematic viscosity

        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)

        # velocity and pressure fields
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))

        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)
        self.p_prime = np.zeros_like(self.p)

        # Rhie-Chow coefficients
        self.d_u = np.zeros_like(self.u)
        self.d_v = np.zeros_like(self.v)

        # Relaxation
        self.alpha_u = 0.7
        self.alpha_v = 0.7
        self.alpha_p = 0.3

        # Geometry mask (solid = 1)
        self.solid = np.zeros((ny, nx))
        self.geometry = geometry
        self.apply_geometry()

        # Boundary conditions
        self.apply_boundary_conditions()

    # ====================================================
    # GEOMETRIES FROM ASSIGNMENT
    # ====================================================
    def apply_geometry(self):
        if self.geometry == "cavity":
            return

        if self.geometry == "cavity_step":
            lx = int(self.nx / 3)
            ly = int(self.ny / 3)
            self.solid[:ly, :lx] = 1

        if self.geometry == "backstep":
            h = int(self.ny / 2)
            self.solid[:h, :] = 1

    # ====================================================
    # BOUNDARY CONDITIONS
    # ====================================================
    def apply_boundary_conditions(self):
        # Lid-driven cavity
        if self.geometry in ["cavity", "cavity_step"]:
            self.u[-1, :] = 1.0  # top lid

        # Back-step inlet
        if self.geometry == "backstep":
            self.u[:, 0] = 1.0   # inlet
            self.v[:, 0] = 0.0

    # ====================================================
    # MOMENTUM EQUATION SOLVER
    # ====================================================
    def solve_u_momentum(self):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        N = nx * ny
        A = lil_matrix((N, N))
        b = np.zeros(N)

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if self.solid[j, i] == 1:
                    continue

                idx = j * nx + i

                u = self.u[j, i]
                v = self.v[j, i]

                # Advection terms
                ue = (self.u[j, i] + self.u[j, i + 1]) / 2
                uw = (self.u[j, i] + self.u[j, i - 1]) / 2

                vn = (self.v[j, i] + self.v[j + 1, i]) / 2
                vs = (self.v[j, i] + self.v[j - 1, i]) / 2

                # Diffusion
                ae = self.nu / dx**2
                aw = self.nu / dx**2
                an = self.nu / dy**2
                as_ = self.nu / dy**2

                aP = ae + aw + an + as_ + (ue - uw)/dx + (vn - vs)/dy

                # Fill matrix
                A[idx, idx] = aP

                A[idx, idx + 1] = -ae
                A[idx, idx - 1] = -aw
                A[idx, idx + nx] = -an
                A[idx, idx - nx] = -as_

                # Pressure gradient term
                dpdx = (self.p[j, i + 1] - self.p[j, i - 1]) / (2 * dx)
                b[idx] = -dpdx + (1 - self.alpha_u) * aP * u

                # Rhie-Chow correction coefficient
                self.d_u[j, i] = dx**2 / aP

        # Solve linear system
        U = spsolve(A.tocsr(), b)

        # Map back into 2D array
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                self.u_star[j, i] = U[idx]

    def solve_v_momentum(self):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        N = nx * ny
        A = lil_matrix((N, N))
        b = np.zeros(N)

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if self.solid[j, i] == 1:
                    continue

                idx = j * nx + i
                u = self.u[j, i]
                v = self.v[j, i]

                vn = (v + self.v[j + 1, i]) / 2
                vs = (v + self.v[j - 1, i]) / 2

                ue = (self.u[j, i] + self.u[j, i + 1]) / 2
                uw = (self.u[j, i] + self.u[j, i - 1]) / 2

                ae = self.nu / dx**2
                aw = self.nu / dx**2
                an = self.nu / dy**2
                as_ = self.nu / dy**2

                aP = ae + aw + an + as_ + (ue - uw)/dx + (vn - vs)/dy
                A[idx, idx] = aP

                A[idx, idx + 1] = -ae
                A[idx, idx - 1] = -aw
                A[idx, idx + nx] = -an
                A[idx, idx - nx] = -as_

                dpdy = (self.p[j + 1, i] - self.p[j - 1, i]) / (2 * dy)
                b[idx] = -dpdy + (1 - self.alpha_v) * aP * v

                self.d_v[j, i] = dy**2 / aP

        V = spsolve(A.tocsr(), b)

        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                self.v_star[j, i] = V[idx]

    # ====================================================
    # PRESSURE POISSON (CORRECTION)
    # ====================================================
    def solve_pressure(self):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        N = nx * ny
        A = lil_matrix((N, N))
        b = np.zeros(N)

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if self.solid[j, i] == 1:
                    continue

                idx = j * nx + i

                A[idx, idx] = -2*(1/dx**2 + 1/dy**2)
                A[idx, idx + 1] = 1/dx**2
                A[idx, idx - 1] = 1/dx**2
                A[idx, idx + nx] = 1/dy**2
                A[idx, idx - nx] = 1/dy**2

                du_dx = (self.u_star[j, i + 1] - self.u_star[j, i - 1]) / (2 * dx)
                dv_dy = (self.v_star[j + 1, i] - self.v_star[j - 1, i]) / (2 * dy)

                b[idx] = -(du_dx + dv_dy)

        P = spsolve(A.tocsr(), b)
        self.p_prime = P.reshape((ny, nx))

        self.p += self.alpha_p * self.p_prime

    # ====================================================
    # VELOCITY CORRECTION
    # ====================================================
    def correct_velocities(self):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                self.u[j, i] = self.u_star[j, i] - self.d_u[j, i] * (self.p_prime[j, i + 1] - self.p_prime[j, i - 1])/(2*dx)
                self.v[j, i] = self.v_star[j, i] - self.d_v[j, i] * (self.p_prime[j + 1, i] - self.p_prime[j - 1, i])/(2*dy)

    # ====================================================
    # MAIN SIMPLE LOOP
    # ====================================================
    def solve(self, max_iter=5000, tol=1e-6):
        for it in range(max_iter):
            self.solve_u_momentum()
            self.solve_v_momentum()
            self.solve_pressure()
            self.correct_velocities()

            # Compute divergence
            div = np.max(np.abs(
                (self.u[:, 2:] - self.u[:, :-2])/(2*self.dx) +
                (self.v[2:, :] - self.v[:-2, :])/(2*self.dy)
            ))

            if it % 50 == 0:
                print(f"Iter {it}: div = {div:.3e}")

            if div < tol:
                print("Converged!")
                break

    # ====================================================
    # PLOTTING UTILITIES
    # ====================================================
    def plot_streamlines(self):
        x = np.linspace(0, 1, self.nx)
        y = np.linspace(0, 1, self.ny)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(6, 6))
        plt.streamplot(X, Y, self.u, self.v, density=2)
        plt.title(f"Streamlines ({self.geometry})")
        plt.gca().invert_yaxis()
        plt.show()

    def plot_pressure(self):
        plt.figure(figsize=(6, 6))
        plt.contourf(self.p, levels=50, cmap="jet")
        plt.colorbar()
        plt.title(f"Pressure field ({self.geometry})")
        plt.gca().invert_yaxis()
        plt.show()


# ====================================================
# RUN CASES (AS REQUIRED BY ASSIGNMENT)
# ====================================================
if __name__ == "__main__":

    # Example: Lid-driven cavity, Re=100, 257×257 grid
    solver = SimpleSolver(nx=257, ny=257, Re=100, geometry="cavity")
    solver.solve(max_iter=2000)
    solver.plot_streamlines()
    solver.plot_pressure()

    # Example: Step cavity, Re=200
    solver = SimpleSolver(nx=320, ny=320, Re=200, geometry="cavity_step")
    solver.solve(max_iter=3000)
    solver.plot_streamlines()
    solver.plot_pressure()

    # BONUS: Back-step flow
    solver = SimpleSolver(nx=320, ny=160, Re=200, geometry="backstep")
    solver.solve(max_iter=3000)
    solver.plot_streamlines()
    solver.plot_pressure()
