import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os
import time


gamma = 20.0 # Thermal Conductivity Coefficient
h_f = 10.0 # Heat transfer coefficient
phi_left = 10.0 # Left Side Temperature (x=0)
phi_right = 100.0 # Right Side Temperature (x=L)
phi_ext = 300.0 # External Temperature (at top of CV)
meshes = [(80,80), (160,160), (320,320)] # 3 Mesh Sizes, Coarse, Medium, Fine. Used as N values
Lx, Ly = 1.0, 1.0 # Set the size of the box to be square, 1 unit length





def build_grid (Nx, Ny, Rx, Ry): # L = Lengths, N = Number of Nodes

    def spacing_1d(L, N, r):
        if abs(r-1.0) < 1e-12: # If inflation value are 1, no skewing of grid.
            dx = np.full(N, L/N) # Grid is set up as N nodes, initial values set to be L/N (1/80 for ex)
            return dx
        N_half = N // 2 # Rounded division
        # Use Denominator of Equation 6. If r very close to 1, assume it is and do same as above
        denom = 1.0 - r ** N_half
        if abs(denom) < 1e-16:
            # r very close to 1; fallback to uniform
            return np.full(N, L / N)
        dx0 = ((1.0 - r) / denom) * (L / 2.0) # Equation 6
        dx_half = dx0 * r ** np.arange(N_half) # First Case in Eqn 7
        dx = np.concatenate([dx_half, dx_half[::-1]]) # Full Eqn 7

        if N % 2 == 1:
            # Insert middle cell
            dx_center = dx_half[-1] * r
            dx = np.concatenate([dx_half, [dx_center], dx_half[::-1]])

        dx *= (L / dx.sum()) # rescaling to ensure sum(dx) = L. Removes error from rounding.
        return dx

    dx = spacing_1d(Lx, Nx, Rx)
    dy = spacing_1d(Ly, Ny, Ry)

    # compute cell centers
    x_nodes = np.zeros(Nx) # Create a blank array for X node centers
    y_nodes = np.zeros(Ny) # Create a blank array for Y node centers
    x_nodes[0] = dx[0] / 2.0
    y_nodes[0] = dy[0] / 2.0

    for i in range (1, Nx):
        x_nodes[i] = x_nodes[i-1] + 0.5 * (dx[i-1] + dx[i])
    for j in range (1, Ny):
        y_nodes[j] = y_nodes[j-1] + 0.5 * (dy[j-1] + dy[j])
    return x_nodes, y_nodes, dx, dy

def idx(i, j, Nx):
    # map 2D (i,j) to 1D index (row-major)
    return j * Nx + i


def assemble_system(Nx, Ny, Rx, Ry):
    # Function creates the A and B matrices to be solved.
    x_c, y_c, dx, dy = build_grid(Nx, Ny, Rx, Ry)
    N = Nx * Ny
    rows, cols, data = [], [], []
    b = np.zeros(N)

    for j in range (Ny):
        for i in range (Nx):
            p = idx(i, j, Nx)
            dxP = dx[i]
            dyP = dy[j]
            aP = 0.0
            # West calculations
            if i == 0:
                # Dirichlet Condition at x=0. Apply backward difference.
                coeff = gamma * dyP / (dxP / 2.0)
                aP += coeff
                b[p] += coeff * phi_left
            else:
                # Interior Face. Apply centered difference
                dxW = 0.5 * (dx[i-1] + dxP)
                Aw = dyP
                aW = gamma * Aw / dxW
                rows.append(p)
                cols.append(idx(i-1, j, Nx))
                data.append(-aW)
                aP += aW
            # East
            if i == Nx - 1:
                # Dirichlet Condition at x=L. Apply backward difference
                coeff = gamma * dyP / (dxP / 2.0)
                aP += coeff
                b[p] += coeff * phi_right
            else:
                dxE = 0.5 * (dx[i+1] + dxP)
                Ae = dyP
                aE = gamma * Ae / dxE
                rows.append(p)
                cols.append(idx(i+1, j, Nx))
                data.append(-aE)
                aP += aE
            # South
            if j == 0:
                # zero gradient as insulating BC. No contribution to matrix or RHS
                pass
            else:
                dyS = 0.5 * (dy[j-1]+dyP)
                As = dxP
                aS = gamma * As / dyS
                rows.append(p)
                cols.append(idx(i, j-1, Nx))
                data.append(-aS)
                aP += aS
            # North
            if j == Ny - 1:
                # Convective Top Boundary Condition - Provided BC
                Af = dxP
                diff_coef = gamma * Af / (dyP / 2.0) # Coeff from diffusion across face
                aP += diff_coef
                conv_coef = h_f * Af
                aP += conv_coef
                b[p] += conv_coef * phi_ext
            else:
                dyN = 0.5 * (dy[j+1] + dyP)
                An = dxP
                aN = gamma * An / dyN
                rows.append(p)
                cols.append(idx(i, j+1, Nx))
                data.append(-aN)
                aP += aN

            rows.append(p)
            cols.append(p)
            data.append(aP)
    A = sp.csr_matrix((data, (rows, cols)), shape=(N,N))
    return A, b, x_c, y_c

def solve_system(A, b):
    phi_vec = spla.spsolve(A, b)
    return phi_vec

def phi_to_grid(phi_vec, Nx, Ny):
    return phi_vec.reshape((Ny, Nx))

def plot_phi(xc, yc, phi, title, fname):
    X, Y = np.meshgrid(xc, yc)
    fig, ax = plt.subplots(figsize=(8,6))
    # levels = np.linspace(phi.min(), phi.max(), 21) # 20 evenly-spaced colour bands over range
    levels = np.linspace (0, 100, 21)
    ticks = np.linspace (0, 100, 11)
    cs = plt.contourf(X, Y, phi, levels=levels)
    cbar = plt.colorbar(cs, ax=ax, ticks = ticks)
    cbar.set_label('Temperature: phi', fontsize = 11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def compute_error_norm(phi_coarse, x_c, y_c, phi_ref, x_ref, y_ref):
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


def run_case(Nx, Ny, Rx, Ry, save_dir='my_results'):
    print(f"\nRunning {Nx}×{Ny} grid (r_x={Rx:.2f}, r_y={Ry:.2f})...")
    os.makedirs(save_dir, exist_ok=True)
    t_start = time.time()
    A, b, xc, yc = assemble_system(Nx, Ny, Rx, Ry)
    t_assembly = time.time() - t_start

    t_start = time.time()
    phi_vec = solve_system(A, b)
    t_solve = time.time() - t_start

    phi = phi_to_grid(phi_vec, Nx, Ny)
    residual = np.linalg.norm(A @ phi_vec - b) # @ does matrix multiplication

    print(f"  Assembly time: {t_assembly:.3f} s")
    print(f"  Solve time: {t_solve:.3f} s")
    print(f"  Residual: {residual:.3e}")
    print(f"  Solution range: [{phi.min():.2f}, {phi.max():.2f}]")

    inflation_str = f"_r{Rx:.2f}" if Rx > 1.0 else ""
    title = f'Temperature Distribution ({Nx}x{Ny})'
    fname = os.path.join(save_dir, f'phi_{Nx}x{Ny}{inflation_str}.png')

    plot_phi(xc, yc, phi, title=title, fname=fname)

    return xc, yc, phi


def convergence_study(Rx, Ry, save_dir='my_results'):
    results = {}
    for Nx, Ny in meshes:
        xc, yc, phi = run_case(Nx, Ny, Rx, Ry, save_dir)
        results[(Nx,Ny)] = (xc,yc,phi)
        print(f'Ran {Nx} x {Ny}')
        print(f'  phi.shape = {phi.shape}')  # ADD THIS
        print(f'  xc.shape = {xc.shape}, yc.shape = {yc.shape}')  # ADD THIS

    finest = meshes[-1]
    xref, yref, phiref = results[finest]
    norms = {}

    Nc = meshes[0]
    Nf = meshes[1]
    xc, yc, phic = results[Nc]
    xf, yf, phif = results[Nf]

    ec = compute_error_norm(phic, xc, yc, phiref, xref, yref)
    ef = compute_error_norm(phif, xf, yf, phiref, xref, yref)

    hx_c = Lx / Nc[0]
    hx_f = Lx / Nf[0]

    print(f"\nErrors:")
    print(f'  ec = {ec:.10e}')
    print(f'  ef = {ef:.10e}')
    print(f'  Ratio ec/ef = {ec / ef:.4f}')
    print(f'  hx_c = {hx_c:.6f}, hx_f = {hx_f:.6f}')
    print(f'  Ratio hx_c/hx_f = {hx_c / hx_f:.4f}')

    order = np.log(ec/ef) / np.log(hx_c/hx_f)
    norms[(Nc, Nf)] = (ec, ef, order)
    print(f"\nCoarse: {Nc[0]}×{Nc[1]}, h = {hx_c:.6f}")
    print(f"Fine:   {Nf[0]}×{Nf[1]}, h = {hx_f:.6f}")
    print(f"Error (coarse): {ec:.6e}")
    print(f"Error (fine):   {ef:.6e}")
    print(f"Order of convergence: {order:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    summary_file = os.path.join(save_dir, 'convergence_summary.txt')

    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CONVERGENCE STUDY RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Domain: {Lx} x {Ly}\n")
        f.write(f"Inflation factors: r_x = {Rx:.4f}, r_y = {Ry:.4f}\n")
        f.write(f"Physical parameters:\n")
        f.write(f"  Gamma = {gamma}\n")
        f.write(f"  h_f = {h_f}\n")
        f.write(f"  phi_left = {phi_left}, phi_right = {phi_right}\n")
        f.write(f"  phi_ext = {phi_ext}\n\n")

        Nc, Nf = meshes[0], meshes[1]
        ec, ef, order = norms[(Nc, Nf)]
        f.write("-" * 70 + "\n")
        f.write(f"Coarse mesh: {Nc[0]} x {Nc[1]}, h = {Lx / Nc[0]:.8f}\n")
        f.write(f"Fine mesh:   {Nf[0]} x {Nf[1]}, h = {Lx / Nf[0]:.8f}\n")
        f.write(f"Error (coarse): {ec:.10e}\n")
        f.write(f"Error (fine):   {ef:.10e}\n")
        f.write(f"Order of convergence: {order:.6f}\n\n")

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {summary_file}")
    print(f"{'=' * 70}")

    return norms

if __name__ == '__main__':
    print("\n" + "#" * 70)
    print("# PART 1: UNIFORM GRID CONVERGENCE STUDY")
    print("#" * 70)
    convergence_study(Rx=1.0, Ry=1.0, save_dir='results_uniform')

    # Inflated grid convergence study
    print("\n" + "#" * 70)
    print("# PART 2: INFLATED GRID CONVERGENCE STUDY (r=1.2)")
    print("#" * 70)
    convergence_study(Rx=1.1, Ry=1.1, save_dir='results_inflated')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nCheck the following directories for results:")
    print("  - results_uniform/")
    print("  - results_inflated/")
    print("=" * 70)



