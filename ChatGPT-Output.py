#!/usr/bin/env python3
"""
sparse_solver.py

Usage examples:
  # Build a tridiagonal test system of size n and solve with direct solver:
  python sparse_solver.py --n 5000 --matrix_type tri --method direct

  # Build a banded system (bandwidth = 3 on each side) and solve iteratively with ILU preconditioner:
  python sparse_solver.py --n 100000 --matrix_type band --half_band 3 --method bicgstab --use_ilu

  # Solve from a Matrix Market file (must provide --mtx path):
  python sparse_solver.py --mtx /path/to/A.mtx --rhs /path/to/b.npy --method gmres

Notes:
 - For extremely large problems (~1e6 unknowns) prefer iterative solvers with a preconditioner.
 - Store A in CSR/CSC (this script uses CSR for arithmetic and ILU compatibility).
"""

import argparse
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
from typing import Tuple

def generate_tridiagonal(n: int, a_diag: float = -1.0, b_diag: float = 2.0, c_diag: float = -1.0, dtype=np.float64) -> sp.csr_matrix:
    """Generate a tridiagonal matrix in CSR format (common for 1D finite-difference).
    A = diag(a_diag, -1) + diag(b_diag, 0) + diag(c_diag, +1)
    """
    main = np.full(n, b_diag, dtype=dtype)
    lower = np.full(n-1, a_diag, dtype=dtype)
    upper = np.full(n-1, c_diag, dtype=dtype)
    A = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format='csr', dtype=dtype)
    return A

def generate_banded(n: int, half_band: int = 3, diag_func=None, dtype=np.float64) -> sp.csr_matrix:
    """Generate a symmetric banded matrix with given half-bandwidth.
    diag_func(k, n) -> values for offset k (k >= 0). Default: simple decreasing diag.
    """
    if half_band < 1:
        raise ValueError("half_band must be >= 1")
    offsets = []
    data = []
    for k in range(-half_band, half_band+1):
        offsets.append(k)
        if diag_func is None:
            # simple pattern: main diag = 2, off-diags decrease with distance
            vals = np.full(n - abs(k), 1.0/(1+abs(k)), dtype=dtype)
            if k == 0:
                vals *= 2.0
        else:
            vals = diag_func(abs(k), n).astype(dtype)
        data.append(vals)
    A = sp.diags(data, offsets=offsets, shape=(n, n), format='csr', dtype=dtype)
    return A

def build_rhs(n: int, kind: str = 'ones', dtype=np.float64) -> np.ndarray:
    if kind == 'ones':
        return np.ones(n, dtype=dtype)
    elif kind == 'rand':
        return np.random.rand(n).astype(dtype)
    elif kind == 'exact_tridiag':
        # Build RHS such that solution x = [1,2,...,n] (for testing)
        x = np.arange(1, n+1, dtype=dtype)
        A = generate_tridiagonal(n)
        return A.dot(x)
    else:
        raise ValueError("Unknown rhs kind")

def read_matrix_market(path: str) -> sp.csr_matrix:
    from scipy.io import mmread
    M = mmread(path)
    if not sp.isspmatrix(M):
        M = sp.csr_matrix(M)
    else:
        M = M.tocsr()
    return M

def solve_sparse(A: sp.csr_matrix, b: np.ndarray, method: str = 'direct', tol: float = 1e-8, maxiter: int = None, use_ilu: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Solve Ax = b.
    method: 'direct' | 'cg' | 'bicgstab' | 'gmres'
    use_ilu: compute an ILU preconditioner (if available) and use as LinearOperator for iterative solvers.
    Returns x and a dict with metadata (time, iterations, residual_norm).
    """
    n = A.shape[0]
    if maxiter is None:
        maxiter = min(10*n, 1000000)

    meta = {}
    start = time.time()

    if method == 'direct':
        # Use direct sparse solver (SuperLU behind the scenes)
        x = spla.spsolve(A, b)
        meta['method'] = 'spsolve'
        meta['iters'] = None
    else:
        # choose solver function
        if method == 'cg':
            solver = spla.cg
        elif method == 'bicgstab':
            solver = spla.bicgstab
        elif method == 'gmres':
            solver = spla.gmres
        else:
            raise ValueError("Unknown method: " + method)

        M = None
        if use_ilu:
            # ILU preconditioner (approximate LU). Works best on CSC or CSR; spilu expects csc/csr
            try:
                # choose float64
                ilu_start = time.time()
                # spilu can be memory costly; drop_tol and fill_factor can be tuned
                ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10)
                ilu_time = time.time() - ilu_start
                Mx = lambda x: ilu.solve(x)
                M = spla.LinearOperator(dtype=A.dtype, shape=A.shape, matvec=Mx)
                meta['ilu_time'] = ilu_time
                meta['ilu_present'] = True
            except Exception as e:
                meta['ilu_present'] = False
                meta['ilu_error'] = str(e)
                M = None

        # call iterative solver
        x, info = solver(A, b, tol=tol, maxiter=maxiter, M=M)
        meta['method'] = method
        meta['iters'] = None if info == 0 else info  # solver-specific meaning

    elapsed = time.time() - start
    meta['time'] = elapsed

    # compute residual
    r = A.dot(x) - b
    residual_norm = np.linalg.norm(r)
    meta['residual_norm'] = residual_norm

    return x, meta

def main(argv=None):
    parser = argparse.ArgumentParser(description="Sparse linear system solver (CSR) with direct/iterative options")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--mtx', type=str, help='Path to Matrix Market (.mtx) file for A')
    parser.add_argument('--rhs', type=str, help='Path to numpy .npy file for b (if not provided uses generated b)')
    parser.add_argument('--n', type=int, default=5000, help='Matrix size (if not using --mtx)')
    parser.add_argument('--matrix_type', choices=['tri', 'band'], default='tri', help='Type of synthetic matrix to build')
    parser.add_argument('--half_band', type=int, default=1, help='Half-bandwidth (for band matrix). half_band=1 -> tridiagonal.')
    parser.add_argument('--method', choices=['direct', 'cg', 'bicgstab', 'gmres'], default='direct', help='Solver method')
    parser.add_argument('--use_ilu', action='store_true', help='Use ILU preconditioner for iterative methods')
    parser.add_argument('--tol', type=float, default=1e-8, help='Tolerance for iterative solver')
    parser.add_argument('--seed', type=int, default=12345, help='RNG seed for reproducibility')
    parser.add_argument('--rhs_kind', choices=['ones', 'rand', 'exact_tridiag'], default='ones', help='Which RHS to generate if not reading file')

    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    if args.mtx:
        print(f"Reading matrix from {args.mtx} ...")
        A = read_matrix_market(args.mtx)
        n = A.shape[0]
        if args.rhs:
            b = np.load(args.rhs)
            if b.shape[0] != n:
                raise ValueError("b length does not match A shape")
        else:
            b = build_rhs(n, kind=args.rhs_kind)
    else:
        n = args.n
        print(f"Building synthetic matrix: type={args.matrix_type}, n={n}, half_band={args.half_band}")
        if args.matrix_type == 'tri' or args.half_band == 1:
            A = generate_tridiagonal(n)
        else:
            A = generate_banded(n, half_band=args.half_band)
        b = build_rhs(n, kind=args.rhs_kind)

    print("Matrix info: shape =", A.shape, ", nnz =", A.nnz, ", dtype =", A.dtype)
    print("RHS norm:", np.linalg.norm(b))

    # Solve
    print(f"Solving with method={args.method}, use_ilu={args.use_ilu}, tol={args.tol} ... (this may take time for large n)")
    x, meta = solve_sparse(A, b, method=args.method, tol=args.tol, use_ilu=args.use_ilu)

    print("\n=== SOLVER SUMMARY ===")
    for k, v in meta.items():
        print(f"{k:20s} : {v}")
    print(f"Solution vector sample (first 10 entries): {x[:10]}")
    print("=======================")

if __name__ == "__main__":
    main()
