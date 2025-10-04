"""
assignment2_fvm.py

Finite-volume solver for Gamma * Laplacian(phi) = 0 with mixed BCs.

Requirements:
- Python 3.8+
- numpy, scipy, matplotlib

Usage examples:
    python assignment2_fvm.py --run-all
    python assignment2_fvm.py --nx 80 --ny 80 --inflation 1.0
    python assignment2_fvm.py --nx 160 --ny 160 --inflation 1.05 --solver bicgstab
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import interpolate
import matplotlib.pyplot as plt
import argparse
import os
import time

# Physical constants (from assignment)
GAMMA = 20.0
h_f = 10.0
phi_left = 10.0
phi_right = 100.0
phi_ext = 300.0


def build_grid(Lx, Ly, Nx, Ny, r_x=1.0, r_y=1.0):
    """
    Build computational grid with optional inflation near boundaries.

    Parameters:
    -----------
    Lx, Ly : float
        Domain dimensions
    Nx, Ny : int
        Number of cells
    r_x, r_y : float
        Inflation factors (>1.0 for boundary refinement)

    Returns:
    --------
    x_centers, y_centers : ndarray
        Cell center coordinates
    dx_array, dy_array : ndarray
        Cell sizes in each direction
    """

    def spacing_1d(L, N, r):
        """Compute cell spacings with symmetric inflation."""
        if abs(r - 1.0) < 1e-12:
            return np.full(N, L / N)

        N_half = N // 2

        # Handle odd N by using floor division
        if N % 2 == 1:
            N_half = (N - 1) // 2

        # Formula from assignment: dx0 = ((1 - r) / (1 - r^(N/2))) * L/2
        denom = 1.0 - r ** N_half
        if abs(denom) < 1e-14:
            return np.full(N, L / N)

        dx0 = ((1.0 - r) / denom) * (L / 2.0)

        # Create spacing array: small to large in first half
        dx_first_half = dx0 * (r ** np.arange(N_half))

        # Mirror for second half: large to small
        dx_second_half = dx_first_half[::-1]

        if N % 2 == 1:
            # For odd N, add center cell
            dx_center = dx_first_half[-1] * r
            dx = np.concatenate([dx_first_half, [dx_center], dx_second_half])
        else:
            dx = np.concatenate([dx_first_half, dx_second_half])

        # Rescale to exactly match domain length
        dx *= (L / dx.sum())
        return dx

    dx = spacing_1d(Lx, Nx, r_x)
    dy = spacing_1d(Ly, Ny, r_y)

    # Compute cell centers
    x_centers = np.zeros(Nx)
    y_centers = np.zeros(Ny)

    x_centers[0] = dx[0] / 2.0
    for i in range(1, Nx):
        x_centers[i] = x_centers[i - 1] + 0.5 * (dx[i - 1] + dx[i])

    y_centers[0] = dy[0] / 2.0
    for j in range(1, Ny):
        y_centers[j] = y_centers[j - 1] + 0.5 * (dy[j - 1] + dy[j])

    return x_centers, y_centers, dx, dy


def idx(i, j, Nx):
    """Map 2D cell indices (i,j) to 1D index (row-major ordering)."""
    return j * Nx + i


def assemble_system(Lx, Ly, Nx, Ny, r_x=1.0, r_y=1.0):
    """
    Assemble the sparse linear system A*phi = b using finite volume method.

    Boundary conditions:
    - West (x=0): Dirichlet, phi = phi_left
    - East (x=Lx): Dirichlet, phi = phi_right
    - South (y=0): Neumann, dphi/dy = 0 (insulated)
    - North (y=Ly): Robin, -Gamma*dphi/dy = h_f*(phi_ext - phi)

    Returns:
    --------
    A : sparse matrix
        System matrix
    b : ndarray
        Right-hand side vector
    x_centers, y_centers : ndarray
        Cell center coordinates
    """
    x_c, y_c, dx_arr, dy_arr = build_grid(Lx, Ly, Nx, Ny, r_x, r_y)
    N = Nx * Ny

    # Use lists for efficient sparse matrix construction
    rows, cols, data = [], [], []
    b = np.zeros(N)

    for j in range(Ny):
        for i in range(Nx):
            p = idx(i, j, Nx)
            dxP = dx_arr[i]
            dyP = dy_arr[j]
            aP = 0.0

            # --- WEST FACE ---
            if i == 0:
                # Dirichlet BC: phi_P = phi_left
                # Use backward difference approximation
                # Flux: F_w = -Gamma * (phi_P - phi_left) / (dx/2) * dy
                coeff = GAMMA * dyP / (dxP / 2.0)
                aP += coeff
                b[p] += coeff * phi_left
            else:
                # Interior face: centered difference
                # Distance between cell centers
                dx_face = 0.5 * (dx_arr[i - 1] + dxP)
                face_area = dyP
                coeff = GAMMA * face_area / dx_face
                rows.append(p)
                cols.append(idx(i - 1, j, Nx))
                data.append(-coeff)
                aP += coeff

            # --- EAST FACE ---
            if i == Nx - 1:
                # Dirichlet BC: phi_P = phi_right
                # Flux: F_e = -Gamma * (phi_right - phi_P) / (dx/2) * dy
                coeff = GAMMA * dyP / (dxP / 2.0)
                aP += coeff
                b[p] += coeff * phi_right
            else:
                # Interior face: centered difference
                dx_face = 0.5 * (dx_arr[i + 1] + dxP)
                face_area = dyP
                coeff = GAMMA * face_area / dx_face
                rows.append(p)
                cols.append(idx(i + 1, j, Nx))
                data.append(-coeff)
                aP += coeff

            # --- SOUTH FACE ---
            if j == 0:
                # Neumann BC: dphi/dy = 0
                # This means phi_ghost = phi_P, so F_s = 0
                # No contribution to matrix or RHS
                pass
            else:
                # Interior face: centered difference
                dy_face = 0.5 * (dy_arr[j - 1] + dyP)
                face_area = dxP
                coeff = GAMMA * face_area / dy_face
                rows.append(p)
                cols.append(idx(i, j - 1, Nx))
                data.append(-coeff)
                aP += coeff

            # --- NORTH FACE ---
            if j == Ny - 1:
                # Robin BC: -Gamma * dphi/dy = h_f * (phi_ext - phi)
                # Using backward difference: -Gamma * (phi_ghost - phi_P) / (dy/2) = h_f * (phi_ext - phi_P)
                # Rearranging: phi_ghost = phi_P - (dy/2) * h_f / Gamma * (phi_ext - phi_P)
                # Flux: F_n = -Gamma * (phi_ghost - phi_P) / (dy/2) * dx
                #           = -Gamma * (- (dy/2) * h_f / Gamma * (phi_ext - phi_P)) / (dy/2) * dx
                #           = h_f * (phi_ext - phi_P) * dx
                # Contribution to aP: h_f * dx
                # Contribution to b: h_f * dx * phi_ext
                # PLUS diffusion term: Gamma * (phi_N_ghost - phi_P) / (dy/2) * dx
                face_area = dxP
                dy_face = dyP / 2.0  # Distance to boundary

                # Diffusion coefficient
                coeff_diff = GAMMA * face_area / dy_face
                aP += coeff_diff

                # Convection coefficient
                coeff_conv = h_f * face_area
                aP += coeff_conv
                b[p] += coeff_conv * phi_ext
            else:
                # Interior face: centered difference
                dy_face = 0.5 * (dy_arr[j + 1] + dyP)
                face_area = dxP
                coeff = GAMMA * face_area / dy_face
                rows.append(p)
                cols.append(idx(i, j + 1, Nx))
                data.append(-coeff)
                aP += coeff

            # Add diagonal coefficient
            rows.append(p)
            cols.append(p)
            data.append(aP)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A, b, x_c, y_c


def solve_system(A, b, solver='direct', tol=1e-10, maxiter=2000):
    """
    Solve the linear system A*phi = b.

    Parameters:
    -----------
    A : sparse matrix
        System matrix
    b : ndarray
        Right-hand side
    solver : str
        'direct', 'bicgstab', or 'gmres'
    tol : float
        Convergence tolerance for iterative solvers
    maxiter : int
        Maximum iterations for iterative solvers

    Returns:
    --------
    phi_vec : ndarray
        Solution vector
    info : dict
        Solver information (residual, iterations, etc.)
    """
    info = {}
    start_time = time.time()

    if solver == 'direct':
        phi_vec = spla.spsolve(A, b)
        info['method'] = 'direct'
        info['success'] = True
    elif solver == 'bicgstab':
        phi0 = np.zeros_like(b)
        phi_vec, flag = spla.bicgstab(A, b, x0=phi0, tol=tol, maxiter=maxiter)
        info['method'] = 'bicgstab'
        info['flag'] = flag
        info['success'] = (flag == 0)
        if flag != 0:
            print(f"Warning: BiCGSTAB did not converge, flag={flag}")
    elif solver == 'gmres':
        phi0 = np.zeros_like(b)
        phi_vec, flag = spla.gmres(A, b, x0=phi0, tol=tol, maxiter=maxiter, restart=50)
        info['method'] = 'gmres'
        info['flag'] = flag
        info['success'] = (flag == 0)
        if flag != 0:
            print(f"Warning: GMRES did not converge, flag={flag}")
    else:
        raise ValueError(f'Unknown solver: {solver}')

    info['time'] = time.time() - start_time
    info['residual'] = np.linalg.norm(A @ phi_vec - b)

    return phi_vec, info


def phi_to_grid(phi_vec, Nx, Ny):
    """Reshape 1D solution vector to 2D grid."""
    return phi_vec.reshape((Ny, Nx))


def plot_phi(xc, yc, phi, title, fname=None, show=False):
    """
    Create contour plot of solution.

    Parameters:
    -----------
    xc, yc : ndarray
        Cell center coordinates
    phi : ndarray (Ny x Nx)
        Solution field
    title : str
        Plot title
    fname : str, optional
        Filename to save plot
    show : bool
        Whether to display plot interactively
    """
    X, Y = np.meshgrid(xc, yc)

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.linspace(phi.min(), phi.max(), 21)
    cs = ax.contourf(X, Y, phi, levels=levels, cmap='hot')
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('φ (Temperature)', fontsize=11)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add boundary condition annotations
    ax.text(0.02, 0.5, f'φ={phi_left}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='center', fontsize=9)
    ax.text(0.98, 0.5, f'φ={phi_right}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='center', horizontalalignment='right', fontsize=9)

    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  Plot saved: {fname}")

    if show:
        plt.show()
    else:
        plt.close()


def compute_error_norm(phi_coarse, x_c, y_c, phi_ref, x_ref, y_ref):
    """
    Compute normalized L2 error norm between solutions on different grids.

    Uses interpolation to compare solutions at coincident points.

    Parameters:
    -----------
    phi_coarse : ndarray
        Solution on coarse grid
    x_c, y_c : ndarray
        Coarse grid coordinates
    phi_ref : ndarray
        Reference solution (fine grid)
    x_ref, y_ref : ndarray
        Reference grid coordinates

    Returns:
    --------
    error : float
        Normalized L2 error norm
    """
    # Interpolate reference solution onto coarse grid
    f = interpolate.interp2d(x_ref, y_ref, phi_ref, kind='cubic')
    phi_ref_interp = f(x_c, y_c)

    # Compute normalized L2 norm
    diff = phi_ref_interp - phi_coarse
    error = np.sqrt(np.mean(diff ** 2))

    return error


def run_case(Lx, Ly, Nx, Ny, r_x=1.0, r_y=1.0, solver='direct',
             save=True, show=False, outdir='results'):
    """
    Run a single case: assemble, solve, and visualize.

    Parameters:
    -----------
    Lx, Ly : float
        Domain dimensions
    Nx, Ny : int
        Grid resolution
    r_x, r_y : float
        Inflation factors
    solver : str
        Solver type
    save : bool
        Whether to save results
    show : bool
        Whether to show plots
    outdir : str
        Output directory

    Returns:
    --------
    xc, yc : ndarray
        Grid coordinates
    phi : ndarray
        Solution field
    info : dict
        Solver information
    """
    print(f"\n{'=' * 60}")
    print(f"Running case: {Nx} x {Ny}, r_x={r_x:.3f}, r_y={r_y:.3f}")
    print(f"{'=' * 60}")

    # Assemble system
    print("  Assembling system...")
    A, b, xc, yc = assemble_system(Lx, Ly, Nx, Ny, r_x, r_y)
    print(f"  System size: {A.shape[0]} unknowns, {A.nnz} non-zeros")

    # Solve
    print(f"  Solving with {solver}...")
    phi_vec, info = solve_system(A, b, solver=solver)
    phi = phi_to_grid(phi_vec, Nx, Ny)

    print(f"  Solve time: {info['time']:.3f} s")
    print(f"  Residual: {info['residual']:.3e}")
    print(f"  Solution range: [{phi.min():.2f}, {phi.max():.2f}]")

    # Save results
    if save:
        os.makedirs(outdir, exist_ok=True)

        # Plot
        inflation_str = f"_r{r_x:.2f}" if r_x > 1.0 else ""
        plot_fname = os.path.join(outdir, f'phi_{Nx}x{Ny}{inflation_str}.png')
        plot_phi(xc, yc, phi,
                 f'Temperature Distribution ({Nx}×{Ny})',
                 fname=plot_fname, show=show)

        # Save numerical data
        np.savez(os.path.join(outdir, f'solution_{Nx}x{Ny}{inflation_str}.npz'),
                 phi=phi, x=xc, y=yc, info=info)

    return xc, yc, phi, info


def convergence_study(Lx=1.0, Ly=1.0, meshes=None, r_x=1.0, r_y=1.0,
                      solver='direct', outdir='results'):
    """
    Perform convergence study on multiple mesh resolutions.

    Parameters:
    -----------
    Lx, Ly : float
        Domain dimensions
    meshes : list of tuples
        List of (Nx, Ny) mesh resolutions
    r_x, r_y : float
        Inflation factors
    solver : str
        Solver type
    outdir : str
        Output directory

    Returns:
    --------
    results : dict
        Convergence study results
    """
    if meshes is None:
        meshes = [(80, 80), (160, 160), (320, 320)]

    print(f"\n{'#' * 60}")
    print(f"# CONVERGENCE STUDY")
    print(f"# Meshes: {meshes}")
    print(f"# Inflation: r_x={r_x:.3f}, r_y={r_y:.3f}")
    print(f"{'#' * 60}")

    # Run all cases
    solutions = {}
    for Nx, Ny in meshes:
        xc, yc, phi, info = run_case(Lx, Ly, Nx, Ny, r_x, r_y,
                                     solver=solver, outdir=outdir)
        solutions[(Nx, Ny)] = {'x': xc, 'y': yc, 'phi': phi, 'info': info}

    # Use finest mesh as reference
    finest_key = meshes[-1]
    phi_ref = solutions[finest_key]['phi']
    x_ref = solutions[finest_key]['x']
    y_ref = solutions[finest_key]['y']

    print(f"\n{'=' * 60}")
    print(f"Computing convergence order (reference: {finest_key[0]}×{finest_key[1]})")
    print(f"{'=' * 60}")

    # Compute errors and convergence order
    convergence_results = []
    for i in range(len(meshes) - 1):
        mesh_c = meshes[i]
        mesh_f = meshes[i + 1]

        # Get solutions
        sol_c = solutions[mesh_c]
        sol_f = solutions[mesh_f]

        # Compute errors against reference
        error_c = compute_error_norm(sol_c['phi'], sol_c['x'], sol_c['y'],
                                     phi_ref, x_ref, y_ref)
        error_f = compute_error_norm(sol_f['phi'], sol_f['x'], sol_f['y'],
                                     phi_ref, x_ref, y_ref)

        # Estimate grid spacing (use mean)
        h_c = Lx / mesh_c[0]
        h_f = Lx / mesh_f[0]

        # Compute order of convergence
        if error_f > 1e-14:  # Avoid division by very small numbers
            order = np.log(error_c / error_f) / np.log(h_c / h_f)
        else:
            order = np.nan

        result = {
            'mesh_coarse': mesh_c,
            'mesh_fine': mesh_f,
            'h_coarse': h_c,
            'h_fine': h_f,
            'error_coarse': error_c,
            'error_fine': error_f,
            'order': order
        }
        convergence_results.append(result)

        print(f"\nCoarse: {mesh_c[0]}×{mesh_c[1]}, h={h_c:.6f}")
        print(f"Fine:   {mesh_f[0]}×{mesh_f[1]}, h={h_f:.6f}")
        print(f"Error (coarse): {error_c:.6e}")
        print(f"Error (fine):   {error_f:.6e}")
        print(f"Order of convergence: {order:.3f}")

    # Save summary
    os.makedirs(outdir, exist_ok=True)
    summary_file = os.path.join(outdir, 'convergence_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("CONVERGENCE STUDY RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Domain: {Lx} × {Ly}\n")
        f.write(f"Inflation: r_x={r_x:.4f}, r_y={r_y:.4f}\n")
        f.write(f"Solver: {solver}\n\n")

        for result in convergence_results:
            f.write("-" * 70 + "\n")
            f.write(f"Coarse mesh: {result['mesh_coarse'][0]}×{result['mesh_coarse'][1]}, "
                    f"h={result['h_coarse']:.8f}\n")
            f.write(f"Fine mesh:   {result['mesh_fine'][0]}×{result['mesh_fine'][1]}, "
                    f"h={result['h_fine']:.8f}\n")
            f.write(f"Error (coarse): {result['error_coarse']:.10e}\n")
            f.write(f"Error (fine):   {result['error_fine']:.10e}\n")
            f.write(f"Order of convergence: {result['order']:.6f}\n\n")

    print(f"\nSummary saved to: {summary_file}")

    return convergence_results


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Finite Volume Method solver for diffusion equation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--run-all', action='store_true',
                        help='Run full convergence study')
    parser.add_argument('--nx', type=int, default=160,
                        help='Number of cells in x-direction')
    parser.add_argument('--ny', type=int, default=160,
                        help='Number of cells in y-direction')
    parser.add_argument('--inflation', type=float, default=1.0,
                        help='Grid inflation factor (>1.0 for boundary refinement)')
    parser.add_argument('--solver', type=str, default='direct',
                        choices=['direct', 'bicgstab', 'gmres'],
                        help='Linear solver type')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    parser.add_argument('--Lx', type=float, default=1.0,
                        help='Domain length in x-direction')
    parser.add_argument('--Ly', type=float, default=1.0,
                        help='Domain length in y-direction')

    args = parser.parse_args()

    if args.run_all:
        meshes = [(80, 80), (160, 160), (320, 320)]

        # Uniform grid study
        print("\n" + "#" * 60)
        print("# UNIFORM GRID CONVERGENCE STUDY")
        print("#" * 60)
        convergence_study(args.Lx, args.Ly, meshes, r_x=1.0, r_y=1.0,
                          solver=args.solver, outdir=args.outdir + '_uniform')

        # Inflated grid study
        r_inflation = max(args.inflation, 1.05)
        print("\n" + "#" * 60)
        print(f"# INFLATED GRID CONVERGENCE STUDY (r={r_inflation})")
        print("#" * 60)
        convergence_study(args.Lx, args.Ly, meshes,
                          r_x=r_inflation, r_y=r_inflation,
                          solver=args.solver,
                          outdir=args.outdir + f'_inflated_r{r_inflation:.2f}')
    else:
        # Single case
        run_case(args.Lx, args.Ly, args.nx, args.ny,
                 r_x=args.inflation, r_y=args.inflation,
                 solver=args.solver, save=True, show=args.show,
                 outdir=args.outdir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()