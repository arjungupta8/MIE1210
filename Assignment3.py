import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os

Lx = 1.0
Ly = 1.0
meshes = [(80,80), (160,160), (320,320)]
phiLeft = 100
phiRight = 0
phiTop = 100
phiBottom = 0

def constant_vel (ux, uy):
    def ux_func(x, y):
        return np.full_like(x, ux)
    def uy_func(x, y):
        return np.full_like(x, uy)
    return ux_func, uy_func

def rotational_vel():
    def ux_func(x,y):
        r = np.sqrt((x-Lx/2)**2 + (y-Ly/2)**2)
        theta = np.arctan2(y-Ly/2, x-Lx/2)
        return -r*np.sin(theta)
    def uy_func(x,y):
        r = np.sqrt((x - Lx / 2) ** 2 + (y - Ly / 2) ** 2)
        theta = np.arctan2(y - Ly / 2, x - Lx / 2)
        return r * np.cos(theta)
    return ux_func, uy_func



def build_grid(nx, ny):
    dx = Lx / nx # spacing between points for uniform spacing in x
    dy = Ly / ny # spacing between points for uniform spacing in y

    xc = np.linspace(dx/2, Lx - dx/2, nx) # cell centers in x dir.
    yc = np.linspace(dy/2, Ly - dy/2, ny) # cell centers in y dir.

    return xc, yc, dx, dy

def index(i, j, nx):
    return j*nx + i # Maps 2D cell index to 1D index


def assemble_system(nx, ny, ux_func, uy_func, scheme, gamma):
    #main updates from A2
    xc, yc, dx, dy = build_grid(nx, ny)
    n = nx*ny # total number of nodes
    rows, cols, data = [], [], [] # row = row index of element, cols = column element, data = value
    b = np.zeros(n)

    x, y = np.meshgrid(xc, yc) # creates 2D meshgrid for velocity evaluation from the center locations
    ux_grid = ux_func(x,y) # calculates the velocities at all cell centers
    uy_grid = uy_func(x,y)

    for j in range (ny):
        for i in range (nx):
            p = index(i, j, nx) # Index of current cell
            aP = 0.0 # Coefficient for the center cell P
            on_left = (i==0) # boolean variables to test if the cell is on a boundary, and which boundary
            on_right = (i == nx - 1)
            on_bottom = (j == 0)
            on_top = (j == ny - 1)
            # Boundary Cells
            if on_left:
                rows.append(p)
                cols.append(p)
                data.append(1.0)
                b[p] = phiLeft
                continue
            elif on_right:
                # Dirichlet BC at x = Lx
                rows.append(p)
                cols.append(p)
                data.append(1.0)
                b[p] = phiRight
                continue
            elif on_bottom:
                # Dirichlet BC at y = 0
                rows.append(p)
                cols.append(p)
                data.append(1.0)
                b[p] = phiBottom
                continue
            elif on_top:
                # Dirichlet BC at y = Ly
                rows.append(p)
                cols.append(p)
                data.append(1.0)
                b[p] = phiTop
                continue
            # Interior Cells
            # ux_p = ux_grid[j, i]
            # uy_p = uy_grid[j, i]

            # Diffusion Terms (RHS of the equation)
            # Always uses central differencing as diffusion is symmetric

            aE_diff = gamma / dx ** 2
            rows.append(p)
            cols.append(index(i+1,j,nx))
            data.append(-aE_diff)
            aP += aE_diff

            aW_diff = gamma / dx**2
            rows.append(p)
            cols.append(index(i-1,j,nx))
            data.append(-aW_diff)
            aP += aW_diff

            aN_diff = gamma / dy**2
            rows.append(p)
            cols.append(index(i,j+1,nx))
            data.append(-aN_diff)
            aP += aN_diff

            aS_diff = gamma / dy**2
            rows.append(p)
            cols.append(index(i,j-1,nx))
            data.append(-aS_diff)
            aP += aS_diff

            # Advection Terms (LHS)
            # This uses the different schemes

            ux_e = (ux_grid[j,i] + ux_grid[j,i+1]) / 2.0 # Interpolated x velocity at east face center
            ux_w = (ux_grid[j,i-1] + ux_grid[j,i]) / 2.0
            uy_n = (uy_grid[j,i] + uy_grid[j+1,i]) / 2.0
            uy_s = (uy_grid[j-1,i] + uy_grid[j,i]) / 2.0

            if scheme == 'central':

                # East Face x-direction contribution to advection
                rows.append(p)
                cols.append(index(i+1,j,nx)) # phi_E coefficient
                data.append(ux_e / (2.0 * dx))
                aP += ux_e / (2.0 * dx) # phi_P coefficient from east face

                # West Face x-direction contribution to advection
                rows.append(p)
                cols.append(index(i-1,j,nx)) # phi_W coefficient
                data.append(-ux_w/(2.0 * dx))
                aP -= ux_w/(2.0 * dx) # phi_P coefficient from west face

                # North face y-direction contribution to advection
                rows.append(p)
                cols.append(index(i,j+1,nx)) # phi_N coefficient
                data.append(uy_n/(2.0 * dy))
                aP += uy_n / (2.0 * dy) # phi_P coefficient from north face

                # South face y-direction contribution to advection
                rows.append(p)
                cols.append(index(i,j-1,nx))
                data.append(-uy_s/(2.0 * dy))
                aP -= uy_s / (2.0 * dy)

            elif scheme == 'upwind':
                # East face: flux = ux_e * phi_upwind
                if ux_e > 0:
                    # Flow from P to E, use phi_P
                    aP -= ux_e / dx
                else:
                    # Flow from E to P, use phi_E
                    rows.append(p)
                    cols.append(index(i+1,j,nx))
                    data.append(-ux_e / dx)
                # West face: flux = ux_w * phi_upwind
                if ux_w > 0:
                    # Flow from W to P, use phi_W
                    rows.append(p)
                    cols.append(index(i-1,j,nx))
                    data.append(ux_w / dx)
                else:
                    # Flow from P to W, use phi_P
                    aP += ux_w / dx
                # North face: flux = uy_n*phi_upwind
                if uy_n > 0:
                    # Flow from P to N, use phi_P
                    aP -= uy_n / dy
                else:
                    # Flow from N to P, use phi_N
                    rows.append(p)
                    cols.append(index(i,j-1,nx))
                    data.append(-uy_n/dy)
                # South Face: flux = uy_s * phi_upwind
                if uy_s > 0:
                    # Flow from S to P, use phi_S
                    rows.append(p)
                    cols.append(index(i,j+1,nx))
                    data.append(uy_s/dy)
                else:
                    # Flow from P to S, use phi_P
                    aP += uy_s/dy

            elif scheme == 'quick':
                # Start with standard upwind, then correct with QUICK:
                # Hayase's method. A constains only upwind, correction is added to RHS.
                # Easier to implement, more accurate. Is iterative.
                # East face: flux = ux_e * phi_upwind
                if ux_e > 0:
                    # Flow from P to E, use phi_P
                    aP -= ux_e / dx
                else:
                    # Flow from E to P, use phi_E
                    rows.append(p)
                    cols.append(index(i + 1, j, nx))
                    data.append(-ux_e / dx)
                # West face: flux = ux_w * phi_upwind
                if ux_w > 0:
                    # Flow from W to P, use phi_W
                    rows.append(p)
                    cols.append(index(i - 1, j, nx))
                    data.append(ux_w / dx)
                else:
                    # Flow from P to W, use phi_P
                    aP += ux_w / dx
                # North face: flux = uy_n*phi_upwind
                if uy_n > 0:
                    # Flow from P to N, use phi_P
                    aP -= uy_n / dy
                else:
                    # Flow from N to P, use phi_N
                    rows.append(p)
                    cols.append(index(i, j - 1, nx))
                    data.append(-uy_n / dy)
                # South Face: flux = uy_s * phi_upwind
                if uy_s > 0:
                    # Flow from S to P, use phi_S
                    rows.append(p)
                    cols.append(index(i, j + 1, nx))
                    data.append(uy_s / dy)
                else:
                    # Flow from P to S, use phi_P
                    aP += uy_s / dy

                # Now implement standard QUICK
                # x-direction
                # East face
                if ux_e > 0:
                    # Flow from P to E, reconstruction point is P
                    if i >= 1:
                        rows.append(p)
                        cols.append(index(i-1, j, nx))
                        data.append(ux_e / (8.0 * dx)) # -1/8 from phi_W
                        aP -= (6.0 * ux_e) / (8.0 * dx)  # 6/8 from phi_P
                        rows.append(p)
                        cols.append(index(i+1, j, nx))
                        data.append((3.0 * -ux_e) / (8.0 * dx)) # 3.8 from phi_E
                    else:
                        aP -= ux_e / dx # Boundary Handling, use upwind. Easier, ensures non-negative contributions
                else:
                    # Flow from E to P, reconstruction point is E
                    if i <= nx - 2:
                        rows.append(p)
                        cols.append(index(i+1, j, nx))
                        data.append((-6.0 * ux_e) / (8.0 * dx)) # 6/8 from phi_E
                        aP += (3.0 * ux_e) / (8.0 * dx) # 3/8 from phi_P
                        if i <= nx - 3:
                            # if far enough from boundary, can maintain third-order accuracy
                            rows.append(p)
                            cols.append(index(i+2, j, nx))
                            data.append(ux_e / (8.0 * dx)) # -1/8 from phi_EE
                        else:
                            # Else adjust to second-order for that one node
                            rows.append(p)
                            cols.append(index(i+1, j, nx))
                            data.append((-3.0 * ux_e) / (8.0 * dx))
                    else:
                        # Too close to boundary, use upwind
                        rows.append(p)
                        cols.append(index(i+1, j, nx))
                        data.append(-ux_e / dx)

                # West face
                if ux_w > 0:
                    # Flow from W to P, reconstruction point is W
                    if i >= 2:
                        # Far enough from boundary, use QUICK
                        rows.append(p)
                        cols.append(index(i-1, j, nx))
                        data.append((6.0 * ux_w) / (8.0 * dx)) # 6/8 from phi_W
                        aP -= (3.0 * ux_w) / (8.0 * dx) # 3/8 from phi_P
                        rows.append(p)
                        cols.append(index(i-2, j, nx))
                        data.append(-ux_w / (8.0 * dx)) # -1/8 from phi_WW
                    else:
                        # Too close to boundary, fall back to upwind
                        rows.append(p)
                        cols.append(index(i-1, j, nx))
                        data.append(ux_w / dx)
                else:
                    # Flow from P to W, reconstruction point is P
                    if i >= 1:
                        aP += ux_w * 6.0 / (8.0 * dx) # 6/8 from phi_P
                        rows.append(p)
                        cols.append(index(i-1, j, nx))
                        data.append((-3.0 * ux_w) / (8.0 * dx)) # 3/8 from phi_W

                        if i <= nx-2:
                            rows.append(p)
                            cols.append(index(i+1, j, nx))
                            data.append(ux_w / (8.0 * dx)) # -1/8 from phi_E
                        else:
                            rows.append(p)
                            cols.append(index(i-1, j, nx))
                            data.append((-3.0 * ux_w) / (8.0 * dx))
                    else:
                        aP += ux_w / dx

                # y-direction
                # North Face
                if uy_n > 0:
                    # Flow from P to N, reconstruction point is P
                    if j >= 1:
                        rows.append(p)
                        cols.append(index(i, j-1, nx))
                        data.append(uy_n / (8.0 * dy)) # -1/8 from phi_S
                        aP -= (6.0 * uy_n) / (8.0 * dy) # 6/8 from phi_P
                        rows.append(p)
                        cols.append(index(i, j+1, nx))
                        data.append((-3.0 * uy_n) / (8.0 * dy))
                    else: # Too close to boundary
                        aP -= uy_n / dy
                else:
                    # From from N to P, reconstruction point is N
                    if j <= ny-2:
                        rows.append(p)
                        cols.append(index(i, j+1, nx))
                        data.append((-6.0 * uy_n) / (8.0 * dy)) # 6/8 from phi_N
                        aP += (3.0 * uy_n) / (8.0 * dy)
                        if j <= ny - 3: # too close to boundary, cannot use NN accurately
                            rows.append(p)
                            cols.append(index(i, j+2, nx))
                            data.append(uy_n / (8.0 * dy)) # -1/8 from phi_NN
                        else:
                            rows.append(p)
                            cols.append(index(i, j+1, nx))
                            data.append((-3.0 * uy_n) / (8.0 * dy))
                    else:
                        # Use upwind when too close to the boundary
                        rows.append(p)
                        cols.append(index(i, j+1, nx))
                        data.append(-uy_n / dy)
                if uy_s > 0:
                    # Flow from S to P, reconstruction point is S
                    if j >= 2:
                        rows.append(p)
                        cols.append(index(i, j-1, nx))
                        data.append((6.0 * uy_s) / (8.0 * dy)) # 6/8 from phi_S
                        aP -= (3.0 * uy_s) / (8.0 * dy) # 3/8 from phi_P
                        rows.append(p)
                        cols.append(index(i, j-2, nx))
                        data.append(-uy_s / (8.0 * dy)) # from -1/8 from phi_SS
                    else:
                        rows.append(p)
                        cols.append(index(i, j-1, nx))
                        data.append(uy_s / dy)
                else:
                    # Flow from P to S, reconstruction point is P
                    if j >= 1:
                        aP += (6.0 * uy_s) / (8.0 * dy) # 6/8 from phi_P
                        rows.append(p)
                        cols.append(index(i, j-1, nx))
                        data.append((-3.0 * uy_s) / (8.0 * dy)) # 3/8 from phi_S

                        if j <= ny - 2:
                            rows.append(p)
                            cols.append(index(i, j+1, nx))
                            data.append(uy_s / (8.0 * dy)) # -1/8 from phi_N
                        else:
                            # Adjust near boundary
                            rows.append(p)
                            cols.append(index(i, j-1, nx))
                            data.append((-3.0 * uy_s) / (8.0 * dy))
                    else:
                        aP += uy_s / dy

            # Add diagonal coefficient
            rows.append(p)
            cols.append(p)
            data.append(aP)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n,n))
    return A, b, xc, yc

def solve_system(A, b):
    # exact same function as A2
    phi_vec = spla.spsolve(A, b)
    return phi_vec

def phi_to_grid(phi_vec, nx, ny):
    # same function as A2
    return phi_vec.reshape((ny,nx)) # Converts 1D solution vector to 2D grid

def compute_quick_correction(phi, nx, ny, dx, dy, ux_grid, uy_grid):
    # Compute Quick scheme correction term with Van Leer limiter.
    correction = np.zeros(nx * ny)
    for j in range(ny):
        for i in range(nx):
            if i == 0 or i == nx-1 or j==0 or j == ny-1:
                # Skip Boundary cells
                continue

            p = index(i, j, nx)
            corr = 0.0

            # Interpolate velocities to faces
            ux_e = (ux_grid[j,i] + ux_grid[j, i+1]) / 2.0
            ux_w = (ux_grid[j, i-1] + ux_grid[j, i]) / 2.0
            uy_n = (uy_grid[j, i] + uy_grid[j+1, i]) / 2.0
            uy_s = (uy_grid[j-1, i] + uy_grid[j, i]) / 2.0

            # East Face
            if ux_e > 0:
                # Flow from W to P to E (upwind is P)
                if i >= 1 and i < nx-1:
                    phi_UU = phi[j, i-1] # West cell - far upwind
                    phi_U = phi[j, i] # Current Cell (upwind)
                    phi_D = phi[j, i+1] # East cell, downwind

                    # compute gradient ratio r: r = (phi_U - phi_UU) / (phi_D - phi_U)
                    denom = phi_D - phi_U
                    if abs(denom) > 1e-12: # basically if not zero, so no inf errors
                        r = (phi_U - phi_UU) / denom

                        # calculate van leer limiter:
                        psi = (r + abs(r)) / (1.0 + r) # 5.10.4 in Versteeg
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr -= ux_e * quick_correction / dx
            else:
                # Flow from E to P to W (upwind is E)
                if i >= 0 and i < nx - 1:
                    phi_U = phi[j, i+1] # East cell - upwind
                    phi_D = phi[j,i] # current cell - downwind
                    if i < nx - 2:
                        phi_UU = phi[j, i+2] # far east - far upwind
                    else:
                        phi_UU = phi_U # Use standard upwind if no far neighbour available
                    denom = phi_D- phi_U
                    if abs(denom) > 1e-12:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + r)
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr -= ux_e * quick_correction / dx
            # West Face
            if ux_w > 0:
                # Flow is from WW to W to P (upwind is W)
                if i >= 2 and i < nx:
                    # Far enough from Boundary
                    phi_UU = phi[j, i-2] # Far west = far upwind
                    phi_U = phi[j, i-1] # West cell = upwind
                    phi_D = phi[j, i] # Current Cell = downwind
                    denom = phi_D - phi_U
                    if abs(denom) > 1e-12:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + r)
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr += ux_w * quick_correction/dx
            else:
                # Flow is from P to W to WW (upwind is P)
                if i >= 1:
                    phi_U = phi[j, i] # Current cell = upwind
                    phi_D = phi[j, i-1] # West cell = downwind
                    if i+1 < nx:
                        # Check that the East Cell Exists
                        phi_UU = phi[j, i+1] # East cell = far upwind
                    else:
                        phi_UU = phi_U # If east cell does not, just do standard upwind
                    denom = phi_D - phi_U
                    if abs(denom) > 1e-12:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + r)
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr += ux_w * quick_correction / dx
            # North Face
            if uy_n > 0:
                # Flow from S to P to N (upwind is P)
                if j >= 1 and j < ny - 1:
                    phi_UU = phi[j-1, i] # South Cell = far upwind
                    phi_U = phi[j, i] # Current Cell = upwind
                    phi_D = phi[j+1, i] # North cell = downwind
                    denom = phi_D - phi_U
                    if abs(denom) > 1e-12:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + r)
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr -= uy_n * quick_correction / dy
            else:
                # Flow is from N to P to S (upwind is N)
                if j>= 0 and j < ny-1:
                    phi_U = phi[j+1, i] # North Cell (upwind)
                    phi_D = phi[j,i]
                    if j < ny-2:
                        # If far enough away from boundary, use far north cell = far upwind
                        phi_UU = phi[j+2, i]
                    else: # Else standard upwind
                        phi_UU = phi_U
                    denom = phi_D - phi_U
                    if abs(denom) > 1e-12:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + r)
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr -= uy_n * quick_correction / dy
            # South Face
            if uy_s > 0:
                # Flow is from SS to S to P (upwind is S)
                if j >= 2:
                    phi_UU = phi[j-2, i] # Far south cell = far upwind
                    phi_U = phi[j-1, i] # South Cell = upwind
                    phi_D = phi[j,i] # Current Cell = downwind
                    denom = phi_D - phi_U
                    if abs(denom) > 1e-12:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + r)
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr += uy_s * quick_correction / dy
            else:
                # Flow is from P to S to SS (upwind is P
                if j >= 1:
                    phi_U = phi[j, i] # Current cell =upwind
                    phi_D = phi[j-1, i]
                    if j < ny-1: # Ensure that the north cell does exist
                        phi_UU = phi[j+1, i]
                    else:
                        phi_UU = phi_U # Otherwise just use standard upwind
                    denom = phi_D - phi_U
                    if abs(denom) > 1e-12:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + abs(r))
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr += uy_s * quick_correction / dy
            correction[p] = corr
    return correction


def solve_with_quick_iterations(nx, ny, ux_func, uy_func, gamma):
    # Solve Quick scheme with iterative deferred correction.
    # Will Assemble upwind matrix A and RHS matrix b,
    # Compute QUICK correction
    # Update RHS: b_new = b + correction
    # Solve A*phi_new = b_new
    # Repeat until converged

    max_iter = 25 # Max number of iterations to solve
    tol = 1e-6 # Tolerance of convergence (if less than max iterations)
    xc, yc, dx, dy = build_grid(nx, ny)
    x, y = np.meshgrid(xc, yc)
    ux_grid = ux_func(x, y)
    uy_grid = uy_func(x, y)
    A, b_base, _, _ = assemble_system(nx, ny, ux_func, uy_func, 'quick', gamma)

    phi_vec = solve_system(A, b_base)
    phi = phi_to_grid(phi_vec, nx, ny)

    for iteration in range(max_iter):
        correction = compute_quick_correction(phi, nx, ny, dx, dy, ux_grid, uy_grid)
        b_corrected = b_base + correction
        phi_vec_new = solve_system(A, b_corrected)
        phi_new = phi_to_grid(phi_vec_new, nx, ny)
        change = np.max(np.abs(phi_new - phi)) # Check convergence
        print (f" QUICK iteration {iteration + 1}: max change = {change:.6e}")
        if change < tol:
            print (f"QUICK converged in {iteration + 1} iterations")
            return xc, yc, phi_new, iteration + 1
        phi = phi_new
        phi_vec = phi_vec_new

    print (f" QUICK reached max iterations ({max_iter}")
    return xc, yc, phi, max_iter



def extract_diagonal_profile(xc, yc, phi, nx, ny):
    x, y = np.meshgrid(xc, yc)
    diag_dist = []
    phi_diag = []
    tolerance = max(Lx / nx, Ly / ny) * 1.5  # Extract values within 1.5 cell widths of diagonal
    for j in range(ny):
        for i in range(nx):
            x_val = x[j,i]
            y_val = y[j,i]
            if abs(x_val - y_val) < tolerance:
                dist = np.sqrt(x_val**2 + y_val**2)
                diag_dist.append(dist)
                phi_diag.append(phi[j,i])
    return diag_dist, phi_diag

def compute_error_norm(phi_coarse, x_c, y_c, phi_ref, x_ref, y_ref):
    # exact same function as A2
    interp_func = RegularGridInterpolator( # interp2d was not working - apparently it is no longer supported
        (y_ref, x_ref),
        phi_ref,
        method = 'cubic', # Cubic Interpolation
        bounds_error = False,
        fill_value=None
    )

    Yc, Xc = np.meshgrid(y_c, x_c, indexing='ij')
    points = np.column_stack([Yc.ravel(), Xc.ravel()])
    phi_ref_on_coarse = interp_func(points).reshape(phi_coarse.shape)
    diff = (phi_ref_on_coarse - phi_coarse)
    error =  np.sqrt(np.mean(diff**2))
    return error


def plot_phi(xc, yc, phi, title, fname):

    # exact same function as A2
    x, y = np.meshgrid(xc, yc)
    fig, ax = plt.subplots(figsize=(8,6))
    levels = np.linspace(phi.min(), phi.max(), 21)
    ticks = np.linspace(0, 100, 11)
    cs = plt.contourf(x, y, phi, levels=levels)
    cbar = plt.colorbar(cs, ax=ax, ticks=ticks)
    cbar.set_label('phi', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison(s_c, phi_c, s_u, phi_u, nx, ny, savedir, case, gamma):
    plt.figure(figsize=(8,6))
    plt.plot(s_c, phi_c, 'b--', label = 'Central Differencing')
    plt.plot(s_u, phi_u, 'g--', label = 'Upwind Scheme')
    plt.xlabel('Distance along diagonal(m)')
    plt.ylabel('Phi')
    plt.legend()
    plt.title(f'Comparison along diagonal, Gamma = {gamma}, {nx}x{ny} grid')
    plt.tight_layout()
    plt.grid(True, alpha=0.3, linestyle='--')
    fname = os.path.join(savedir, f'{case}_central_upwind_comparison.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_vel_field(xc, yc, ux_func, uy_func, savedir, case, scheme, nx, ny):
    x, y = np.meshgrid(xc, yc)
    ux_grid = ux_func(x,y)
    uy_grid = uy_func(x,y)

    fig, ax = plt.subplots(figsize=(8,6))
    skip = len(xc)//12 # Matching the given in the assignment, show 12x12 grid of vel field
    ax.quiver(x[::skip, ::skip], y[::skip, ::skip], ux_grid[::skip, ::skip], uy_grid[::skip, ::skip], color="black", scale=20, width=0.0015, headwidth=4, headlength=5)
    ax.plot([0, Lx], [0, 0], "k-")
    ax.plot([0, Lx], [Ly, Ly], "k-")
    ax.plot([0, 0], [0, Ly], "k-")
    ax.plot([Lx, Lx], [0, Ly], "k-")
    ax.text(-0.08, 0.5, r"$\phi_b = 100$", va="center", rotation=90, fontsize=11)
    ax.text(1.02 * Lx, 0.5, r"$\phi_b = 0$", va="center", rotation=90, fontsize=11)
    ax.text(0.5 * Lx, -0.08, r"$\phi_b = 0$", ha="center", fontsize=11)
    ax.text(0.5 * Lx, 1.02 * Ly, r"$\phi_b = 100$", ha="center", fontsize=11)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title(f'Velocity Field, Scheme = {scheme}, Grid = {nx}x{ny}, Gamma = 5', fontsize=12, fontweight="bold", pad=20)
    ax.grid(True, linestyle="-", alpha=0.3)
    filename = os.path.join(savedir, f'Vel_Field_{case}_{scheme}_{nx}x{ny}.png')
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()





def run_case(nx, ny, ux, uy, scheme, savedir, case, gamma):
    print(f"Running {case}: {nx}x{ny} grid, {scheme} scheme...")
    os.makedirs(savedir, exist_ok=True)

    A, b, xc, yc = assemble_system(nx, ny, ux, uy, scheme, gamma)
    phi_vec = solve_system(A, b)
    phi = phi_to_grid(phi_vec, nx, ny)

    residual = np.linalg.norm (A @ phi_vec - b)
    print (f" Residual: {residual:.3e}")
    print (f" Solution Range:  [{phi.min():.4f}, {phi.max():.4f}]")

    title = f" {case} ({nx}x{ny}, {scheme})"
    filename = os.path.join(savedir, f'{case}_{nx}x{ny}_{scheme}.png')
    plot_phi(xc, yc, phi, title=title, fname=filename) # Don't need to plot for most tests, but oh well.

    return xc, yc, phi


def convergence_study(ux_func, uy_func, scheme, savedir, case, gamma):
    results = {}
    for nx, ny in meshes:
        xc, yc, phi = run_case(nx, ny, ux_func, uy_func, scheme, savedir, case, gamma)
        plot_vel_field(xc, yc, ux_func, uy_func, savedir, case, scheme, nx, ny)
        results[(nx,ny)] = (xc, yc, phi)
    finest = meshes[-1]
    xref, yref, phiref = results[finest]

    Nc = meshes[0]
    Nf = meshes[1]
    xc, yc, phic = results[Nc]
    xf, yf, phif = results[Nf]

    ec = compute_error_norm(phic, xc, yc, phiref, xref, yref)
    ef = compute_error_norm(phif, xf, yf, phiref, xref, yref)

    hx_c = Lx / Nc[0]
    hx_f = Lx / Nf[0]
    order = np.log(ec/ef) / np.log(hx_c/hx_f)
    print(f"\nConvergence Results for {scheme}:")
    print(f"  Coarse mesh: {Nc[0]}×{Nc[1]}, h = {hx_c:.6f}, Error = {ec:.6e}")
    print(f"  Fine mesh:   {Nf[0]}×{Nf[1]}, h = {hx_f:.6f}, Error = {ef:.6e}")
    print(f"  Order of convergence: {order:.4f}")

    return order


if __name__ == "__main__":

    print ("Part 2: Gamma = 0, ux = uy = 2: ")
    gamma = 0.0
    ux_func, uy_func = constant_vel(2.0, 2.0)
    case = f'const_vel_gamma0'
    savedir = f'a3_results_part2'
    nx, ny = 160, 160
    xc_c, yc_c, phi_c = run_case(nx, ny, ux_func, uy_func, 'central', savedir=savedir,case=case, gamma=gamma)
    s_c, phi_diag_c = extract_diagonal_profile(xc_c, yc_c, phi_c, nx, ny)

    xc_u, yc_u, phi_u = run_case(nx, ny, ux_func, uy_func, 'upwind', savedir=savedir, case=case, gamma=gamma)
    s_u, phi_diag_u = extract_diagonal_profile(xc_u, yc_u, phi_u, nx, ny)

    plot_comparison(s_c, phi_diag_c, s_u, phi_diag_u, nx, ny, savedir=savedir, case=case, gamma=gamma)

    print ("Part 3: Gamma = 5, ux = uy = 2: ")
    gamma = 5.0
    # can use the same ux and uy functions
    case = f'const_vel_gamma5'
    savedir = f'a3_results_part3'
    nx, ny = 160, 160
    xc_c, yc_c, phi_c = run_case(nx, ny, ux_func, uy_func, 'central', savedir=savedir,case=case, gamma=gamma)
    s_c, phi_diag_c = extract_diagonal_profile(xc_c, yc_c, phi_c, nx, ny)

    xc_u, yc_u, phi_u = run_case(nx, ny, ux_func, uy_func, 'upwind', savedir=savedir, case=case, gamma=gamma)
    s_u, phi_diag_u = extract_diagonal_profile(xc_u, yc_u, phi_u, nx, ny)

    plot_comparison(s_c, phi_diag_c, s_u, phi_diag_u, nx, ny, savedir=savedir, case=case, gamma=gamma)

    print ("Part 4: Gamma = 5, ux = -rsin(theta), uy = rcos(theta): ")
    #can use the same gamma value
    ux_func, uy_func = rotational_vel()
    case = f'rotational_vel_gamma5'
    savedir = f'a3_results_part4'
    order_central= convergence_study(ux_func, uy_func, 'central', savedir, case, gamma)
    order_upwind= convergence_study(ux_func, uy_func, 'upwind', savedir, case, gamma)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind - Order of Convergence: {order_upwind:.4f}")

    print ("BONUS: Gamma = 5, ux = rsin(theta), uy = rcos(theta), QUICK Scheme: ")
    # can use the same gamma value
    # can use the same velocity functions
    case = f'bonus_rotational_vel_gamma5'
    savedir = f'a3_results_bonus'
    order_quick = convergence_study(ux_func, uy_func, 'quick', savedir, case, gamma)
    print (f" Bonus QUICK - Order of Convergence: {order_quick:.4f}")




























