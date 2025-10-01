
"""
assignment2_fvm.py

Finite-volume solver for Gamma * Laplacian(phi) = 0 with mixed BCs.

Requirements:
- Python 3.8+
- numpy, scipy, matplotlib

Usage examples:
    python assignment2_fvm.py --run-all
    python assignment2_fvm.py --nx 80 --ny 80 --inflation 1.0

The script will produce contour plots and a convergence study for 80x80, 160x160, 320x320.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import interpolate
import matplotlib.pyplot as plt
import argparse
import os

# Physical constants (assignment)
GAMMA = 20.0
h_f = 10.0
phi_left = 10.0
phi_right = 100.0
phi_ext = 300.0

# Helper: build grid (cell centers) with optional inflation

def build_grid(Lx, Ly, Nx, Ny, r_x=1.0, r_y=1.0):
    """Return x_centers, y_centers, dx_array, dy_array

    If r_x or r_y > 1.0, apply inflation symmetric about center.
    """
    def spacing_1d(L, N, r):
        # We want N cell spacings that are symmetric about center. Using formula in assignment
        if abs(r - 1.0) < 1e-12:
            dx = np.full(N, L / N)
            return dx
        # compute first spacing using provided formula for half domain (N assumed even)
        N_half = N // 2
        # formula from assignment: dx0 = ((1 - r) / (1 - r^(N/2))) * L / 2  -- careful with power
        denom = 1.0 - r ** N_half
        if abs(denom) < 1e-16:
            # r very close to 1; fallback to uniform
            return np.full(N, L / N)
        dx0 = ((1.0 - r) / denom) * (L / 2.0)
        dx_half = dx0 * r ** np.arange(N_half)
        dx = np.concatenate([dx_half, dx_half[::-1]])
        # if N odd, insert center cell
        if N % 2 == 1:
            # adjust: create N_half on both sides and one center; use geometric progression until center
            # For simplicity, fallback to uniform when odd
            dx = np.full(N, L / N)
        # rescale to exactly match L due to numerical rounding
        dx *= (L / dx.sum())
        return dx

    dx = spacing_1d(Lx, Nx, r_x)
    dy = spacing_1d(Ly, Ny, r_y)

    # compute cell centers
    x_nodes = np.zeros(Nx)
    x_nodes[0] = dx[0] / 2.0
    for i in range(1, Nx):
        x_nodes[i] = x_nodes[i - 1] + 0.5 * (dx[i - 1] + dx[i])
    y_nodes = np.zeros(Ny)
    y_nodes[0] = dy[0] / 2.0
    for j in range(1, Ny):
        y_nodes[j] = y_nodes[j - 1] + 0.5 * (dy[j - 1] + dy[j])

    return x_nodes, y_nodes, dx, dy


def idx(i, j, Nx):
    # map 2D (i,j) to 1D index (row-major)
    return j * Nx + i


def assemble_system(Lx, Ly, Nx, Ny, r_x=1.0, r_y=1.0):
    x_c, y_c, dx_arr, dy_arr = build_grid(Lx, Ly, Nx, Ny, r_x, r_y)
    N = Nx * Ny
    rows = []
    cols = []
    data = []
    b = np.zeros(N)

    for j in range(Ny):
        for i in range(Nx):
            p = idx(i, j, Nx)
            # local cell sizes
            dxP = dx_arr[i]
            dyP = dy_arr[j]
            aP = 0.0
            # WEST
            if i == 0:
                # Dirichlet at x=0
                rows.append(p); cols.append(p); data.append(1.0)
                b[p] = phi_left
                continue
            else:
                dxW = 0.5 * (dx_arr[i - 1] + dxP)
                Aw = dyP
                aW = GAMMA * Aw / dxW
                rows.append(p); cols.append(idx(i - 1, j, Nx)); data.append(-aW)
                aP += aW
            # EAST
            if i == Nx - 1:
                # Dirichlet at x=Lx
                rows.append(p); cols.append(p); data.append(1.0)
                b[p] = phi_right
                continue
            else:
                dxE = 0.5 * (dx_arr[i + 1] + dxP)
                Ae = dyP
                aE = GAMMA * Ae / dxE
                rows.append(p); cols.append(idx(i + 1, j, Nx)); data.append(-aE)
                aP += aE
            # SOUTH
            if j == 0:
                # zero-gradient: ϕ_S = ϕ_P => Fs = 0  => no contribution
                # Equivalent approach: do nothing (no S neighbor)
                pass
            else:
                dyS = 0.5 * (dy_arr[j - 1] + dyP)
                As = dxP
                aS = GAMMA * As / dyS
                rows.append(p); cols.append(idx(i, j - 1, Nx)); data.append(-aS)
                aP += aS
            # NORTH
            if j == Ny - 1:
                # convective top boundary: implement -Gamma*(phi_ghost - phi_P)/dy_face = h_f*(phi_ext - phi_P)
                # using mirror/ghost approach: -G*(phi_ghost - phi_P)/dy_face = h_f*(phi_ext - phi_P)
                # phi_ghost from this relation => will produce aP increment and b
                dyN = dyP * 0.5  # approximate face distance
                Af = dxP
                # coefficient from diffusion across face: G*A / dy_face
                diff_coef = GAMMA * Af / dyP
                # Combine Robin: diff_coef*(phi_P - phi_ghost) = h_f*(phi_ext - phi_P) with phi_ghost
                # Rearranging yields: aP += diff_coef + h_f * Af; b += h_f * Af * phi_ext
                # But we must ensure consistent dimensions: h_f has units per area; multiply by face area
                aP += diff_coef + h_f * Af
                b[p] += h_f * Af * phi_ext
            else:
                dyN = 0.5 * (dy_arr[j + 1] + dyP)
                An = dxP
                aN = GAMMA * An / dyN
                rows.append(p); cols.append(idx(i, j + 1, Nx)); data.append(-aN)
                aP += aN

            # central coefficient
            rows.append(p); cols.append(p); data.append(aP)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A, b, x_c, y_c


def solve_system(A, b, solver='direct', tol=1e-8, maxiter=1000):
    if solver == 'direct':
        phi_vec = spla.spsolve(A, b)
        return phi_vec
    elif solver == 'bicgstab':
        phi0 = np.zeros_like(b)
        phi_vec, info = spla.bicgstab(A, b, x0=phi0, tol=tol, maxiter=maxiter)
        if info != 0:
            print("Warning: bicgstab did not converge, info=", info)
        return phi_vec
    else:
        raise ValueError('Unknown solver')


def phi_to_grid(phi_vec, Nx, Ny):
    return phi_vec.reshape((Ny, Nx))


def plot_phi(xc, yc, phi, title, fname=None):
    X, Y = np.meshgrid(xc, yc)
    plt.figure(figsize=(6,4.5))
    cs = plt.contourf(X, Y, phi, 20)
    plt.colorbar(cs)
    plt.title(title)
    plt.xlabel('x'); plt.ylabel('y')
    if fname:
        plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()


def compute_error_norm(phi_coarse, x_c, y_c, phi_ref, x_ref, y_ref):
    # Interpolate reference (fine) onto coarse grid and compute normalized L2 norm
    f = interpolate.interp2d(x_ref, y_ref, phi_ref, kind='cubic')
    # Note: interp2d expects 1D arrays sorted ascending
    Xc, Yc = np.meshgrid(x_c, y_c)
    phi_ref_on_coarse = f(x_c, y_c)
    diff = (phi_ref_on_coarse - phi_coarse)
    return np.sqrt(np.mean(diff.ravel()**2))


def run_case(Lx, Ly, Nx, Ny, r_x=1.0, r_y=1.0, solver='direct', save=True, outdir='results'):
    A, b, xc, yc = assemble_system(Lx, Ly, Nx, Ny, r_x, r_y)
    phi_vec = solve_system(A, b, solver=solver)
    phi = phi_to_grid(phi_vec, Nx, Ny)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if save:
        plot_phi(xc, yc, phi, f'phi {Nx}x{Ny}', os.path.join(outdir, f'phi_{Nx}_{Ny}.png'))
        # save numeric solution
        np.save(os.path.join(outdir, f'phi_{Nx}_{Ny}.npy'), phi)
        np.save(os.path.join(outdir, f'xc_{Nx}_{Ny}.npy'), xc)
        np.save(os.path.join(outdir, f'yc_{Nx}_{Ny}.npy'), yc)
    return xc, yc, phi


def convergence_study(Lx=1.0, Ly=1.0, meshes=[(80,80),(160,160),(320,320)], r_x=1.0, r_y=1.0, outdir='results'):
    # compute solutions on given meshes and estimate order using finest as reference
    results = {}
    for Nx, Ny in meshes:
        xc, yc, phi = run_case(Lx, Ly, Nx, Ny, r_x=r_x, r_y=r_y, outdir=outdir)
        results[(Nx,Ny)] = (xc, yc, phi)
        print(f'Ran {Nx} x {Ny}')

    # assume the last mesh is reference (finest)
    finest = meshes[-1]
    xref, yref, phiref = results[finest]

    norms = {}
    for i_mesh in range(len(meshes)-1):
        Nc = meshes[i_mesh]
        Nf = meshes[i_mesh+1]
        xc, yc, phic = results[Nc]
        xf, yf, phif = results[Nf]
        # compute error norms against phiref (interpolate phiref onto coarse and fine grids)
        ec = compute_error_norm(phic, xc, yc, phiref, xref, yref)
        ef = compute_error_norm(phif, xf, yf, phiref, xref, yref)
        # grid spacings: use average dx
        hx_c = np.mean(np.diff(xc))
        hx_f = np.mean(np.diff(xf))
        order = np.log(ec/ef) / np.log(hx_c/hx_f)
        norms[(Nc, Nf)] = (ec, ef, order)
        print(f'coarse {Nc}, fine {Nf}: ec={ec:.3e}, ef={ef:.3e}, order~{order:.3f}')

    # write summary file
    with open(os.path.join(outdir, 'convergence_results.txt'), 'w') as fid:
        fid.write('Convergence study results\n')
        for key, (ec, ef, order) in norms.items():
            fid.write(f'{key}: ec={ec:.6e}, ef={ef:.6e}, order={order:.6f}\n')
    return norms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--nx', type=int, default=80)
    parser.add_argument('--ny', type=int, default=80)
    parser.add_argument('--inflation', type=float, default=1.0)
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    Lx, Ly = 1.0, 1.0
    if args.run_all:
        meshes = [(80,80),(160,160),(320,320)]
        print('Uniform grid convergence study...')
        convergence_study(Lx, Ly, meshes, r_x=1.0, r_y=1.0, outdir=args.outdir)
        print('Inflated grid convergence study (r=1.2)...')
        convergence_study(Lx, Ly, meshes, r_x=args.inflation if args.inflation>1.0 else 1.2, r_y=args.inflation if args.inflation>1.0 else 1.2, outdir=args.outdir+'_inflated')
    else:
        xc, yc, phi = run_case(Lx, Ly, args.nx, args.ny, r_x=args.inflation, r_y=args.inflation, outdir=args.outdir)
        plot_phi(xc, yc, phi, f'phi {args.nx}x{args.ny}', os.path.join(args.outdir, f'phi_{args.nx}_{args.ny}.png'))
