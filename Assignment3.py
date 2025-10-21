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
        return ux
    def uy_func(x, y):
        return uy
    return ux_func, uy_func



def build_grid(nx, ny):
    dx = Lx / nx # spacing between points for uniform spacing in x
    dy = Ly / ny # spacing between points for uniform spacing in y

    xc = np.linspace(dx/2, Lx - dx/2, nx) # cell centers in x dir.
    yc = np.linspace(dy/2, Ly - dy/2, ny) # cell centers in y dir.

    return xc, yc, dx, dy

def index(i, j, nx):
    return j*nx + i # Maps 2D cell index to 1D index


def assemble_system(nx, ny, ux_func, uy_func, scheme):
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
            ux_p = ux_grid[j, i]
            uy_p = uy_grid[j, i]

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
            cols.append(index(i,j+1,ny))
            data.append(-aN_diff)
            aP += aN_diff

            aS_diff = gamma / dy**2
            rows.append(p)
            cols.append(index(i,j-1,ny))
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

                # Add Quick scheme here
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



def run_case(nx, ny, ux, uy, scheme, savedir, case):
    print(f"\nRunning {case}: {nx}x{ny} grid, {scheme} scheme...")
    os.makedirs(savedir, exist_ok=True)

    A, b, xc, yc = assemble_system(nx, ny, ux, uy, scheme)
    phi_vec = solve_system(A, b)
    phi = phi_to_grid(phi_vec, nx, ny)

    residual = np.linalg.norm (A @ phi_vec - b)
    print (f" Residual: {residual:.3e}")
    print (f" Solution Range:  [{phi.min():.4f}, {phi.max():.4f}")

    title = f" {case} ({nx}x{ny}, {scheme}"
    filename = os.path.join(savedir, f'{case}_{nx}x{ny}_{scheme}.png')
    plot_phi(xc, yc, phi, title=title, fname=filename) # Don't need to plot for most tests, but oh well.

    return xc, yc, phi

g

if __name__ = 'main':

    print ("Part 2: Gamma = 0, ux = uy = 2: ")
    gamma = 0.0
    ux_func, uy_func = constant_vel(2.0, 2.0)
    run_case(160, 160, ux_func, uy_func, 'central', 'a3_results_part2','const_vel_gamma0')