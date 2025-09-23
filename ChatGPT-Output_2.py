#!/usr/bin/env python3
"""
sparse_solver_expanded.py

Features:
 - TDMA (Thomas algorithm) for tridiagonal systems (uses only diagonals).
 - SciPy-based solver for general sparse A (direct spsolve or iterative solvers).
 - Auto-detection of tridiagonal structure when loading a sparse matrix.
 - Command-line interface to switch modes, load Matrix Market files, or build synthetic tests.

Usage examples:
  # Tridiagonal with TDMA
  python sparse_solver_expanded.py --n 1000000 --matrix_type tri --method tdma

  # Use SciPy direct on a generated banded matrix
  python sparse_solver_expanded.py --n 10000 --matrix_type band --method direct

  # Load matrix in Matrix Market format and RHS in .npy
  python sparse_solver_expanded.py --mtx A.mtx --rhs b.npy --method bicgstab --use_ilu

Notes:
 - TDMA only applies to strictly tridiagonal matrices (nonzeros only on offsets -1,0,+1).
 - For general sparse problems of size ~1e6, iterative solvers with preconditioning (ILU/AMG) are usually needed.
"""
# --- Imports ---
import argparse                    # parse command-line arguments
import time                        # timing operations
import numpy as np                 # numerical arrays
import scipy.sparse as sp          # sparse matrix structures
import scipy.sparse.linalg as spla # sparse linear algebra solvers
import warnings                    # warn user about potential issues
from typing import Tuple, Optional # type hints for clarity

# --- TDMA (Thomas) solver for tridiagonal systems ---
def tdma_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve tridiagonal system with Thomas algorithm.
    Input arrays:
      a: lower diagonal, shape (n-1,)
      b: main diagonal, shape (n,)
      c: upper diagonal, shape (n-1,)
      d: right-hand side, shape (n,)
    Returns:
      x: solution vector, shape (n,)
    Notes:
      - This function modifies/uses workspace arrays cp, dp of length n (cp length n-1).
      - All arrays should be float64 for numeric stability.
    """
    n = b.size                          # number of unknowns
    if d.size != n:
        raise ValueError("Dimension mismatch: b and d must have same length")
    # allocate forward-modified coefficients
    cp = np.empty(n-1, dtype=b.dtype)   # modified upper-diagonal coefficients
    dp = np.empty(n, dtype=b.dtype)     # modified RHS values
    # first row (i=0): cp0 = c0 / b0 ; dp0 = d0 / b0
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    # forward sweep: eliminate lower-diagonal
    for i in range(1, n):
        denom = b[i] - a[i-1] * cp[i-1]  # pivot after elimination
        # if denom is zero (or near zero) system is singular/ill-conditioned
        if abs(denom) < 1e-16:
            raise ZeroDivisionError(f"Zero (or near-zero) pivot encountered at row {i}")
        if i < n-1:
            cp[i] = c[i] / denom          # compute modified upper diag except last
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom  # modified RHS
    # back substitution
    x = np.empty(n, dtype=b.dtype)
    x[-1] = dp[-1]                      # last unknown
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]   # back-substitute
    return x

# --- Utilities to build test matrices/diagonals ---
def build_tridiagonal_diags(n: int, a_val: float = -1.0, b_val: float = 2.0, c_val: float = -1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return three arrays (a, b, c) representing lower, main, upper diagonals for tridiagonal A.
    a has length n-1, b length n, c length n-1.
    """
    if n < 2:
        raise ValueError("n must be >= 2 for tridiagonal system")
    a = np.full(n-1, a_val, dtype=np.float64)   # sub-diagonal
    b = np.full(n,   b_val, dtype=np.float64)   # main diagonal
    c = np.full(n-1, c_val, dtype=np.float64)   # super-diagonal
    return a, b, c

def build_tridiagonal_sparse(n: int, a_val: float = -1.0, b_val: float = 2.0, c_val: float = -1.0) -> sp.csr_matrix:
    """
    Build a sparse CSR tridiagonal matrix (useful if user wants to call SciPy solvers).
    """
    a, b, c = build_tridiagonal_diags(n, a_val, b_val, c_val)
    # diags expects list of arrays and corresponding offsets
    A = sp.diags([a, b, c], offsets=[-1, 0, 1], format='csr', dtype=np.float64)
    return A

def build_banded_sparse(n: int, half_band: int = 3) -> sp.csr_matrix:
    """
    Build a symmetric banded sparse matrix with half-bandwidth `half_band`.
    This is a synthetic test: main diag = 2, off-diags = 1/(1+distance).
    """
    if half_band < 1:
        raise ValueError("half_band must be >= 1")
    offsets = list(range(-half_band, half_band+1))
    data = []
    for k in offsets:
        # length of diagonal
        length = n - abs(k)
        vals = np.full(length, 1.0/(1+abs(k)), dtype=np.float64)
        if k == 0:
            vals *= 2.0
        data.append(vals)
    A = sp.diags(data, offsets=offsets, shape=(n, n), format='csr', dtype=np.float64)
    return A

# --- Function to read Matrix Market file if user supplies it ---
def read_matrix_market(path: str) -> sp.csr_matrix:
    """
    Read matrix from Matrix Market (.mtx) file and return CSR matrix.
    """
    from scipy.io import mmread
    M = mmread(path)           # read file (may produce dense or sparse)
    if not sp.isspmatrix(M):
        M = sp.csr_matrix(M)   # convert dense -> csr
    else:
        M = M.tocsr()          # ensure CSR format
    return M

# --- Extract diagonals from sparse A if tridiagonal ---
def extract_tridiagonal_from_sparse(A: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a sparse matrix A (csr/csc), extract diagonals a,b,c if A is tridiagonal.
    Raises ValueError if A is not tridiagonal.
    """
    # convert to COOrdinate for easy row/col access (cheap relative to problem size)
    Acoo = A.tocoo()
    rows = Acoo.row
    cols = Acoo.col
    if rows.size == 0:
        raise ValueError("Empty matrix")
    # compute |row - col| for each nonzero entry
    deltas = np.abs(rows - cols)
    max_band = deltas.max()
    if max_band > 1:
        raise ValueError("Matrix is not tridiagonal (max bandwidth = %d)" % int(max_band))
    n = A.shape[0]
    # extract diagonals using .diagonal
    main = A.diagonal()
    lower = A.diagonal(-1)
    upper = A.diagonal(1)
    # ensure lengths: lower/upper are length n-1
    return lower.astype(np.float64), main.astype(np.float64), upper.astype(np.float64)

# --- SciPy-based general sparse solver wrapper ---
def scipy_solve(A: sp.spmatrix, b: np.ndarray, method: str = 'direct', tol: float = 1e-8, maxiter: Optional[int] = None, use_ilu: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Solve Ax = b using SciPy methods.
    method: 'direct' (spsolve) or 'cg'/'bicgstab'/'gmres' (iterative)
    use_ilu: if True, attempt to build an ILU preconditioner for iterative solvers.
    Returns (x, metadata)
    """
    meta = {}
    start = time.time()
    if method == 'direct':
        # direct solve using SuperLU (spsolve)
        x = spla.spsolve(A.tocsr(), b)
        meta['method'] = 'spsolve'
        meta['info'] = 0
    else:
        # pick solver function
        solver = {'cg': spla.cg, 'bicgstab': spla.bicgstab, 'gmres': spla.gmres}.get(method)
        if solver is None:
            raise ValueError("Unknown iterative method: " + method)
        M = None
        if use_ilu:
            # attempt ILU on a copy in CSC format (spilu expects csc/csr)
            try:
                ilu_start = time.time()
                ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10)
                ilu_time = time.time() - ilu_start
                # build LinearOperator using ilu.solve
                M = spla.LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype)
                meta['ilu_time'] = ilu_time
            except Exception as e:
                warnings.warn(f"ILU preconditioner failed: {e}")
                meta['ilu_error'] = str(e)
                M = None
        # run iterative solver
        x, info = solver(A, b, tol=tol, maxiter=maxiter, M=M)
        meta['method'] = method
        meta['info'] = int(info)
    meta['time'] = time.time() - start
    # compute residual
    r = A.dot(x) - b
    meta['residual_norm'] = float(np.linalg.norm(r))
    return x, meta

# --- High-level solve dispatcher that decides between TDMA and SciPy ---
def solve_system(args) -> Tuple[np.ndarray, dict]:
    """
    Given parsed args, build or load A and b, decide solver, and run it.
    Returns x and metadata dict.
    """
    # Build or load A and b
    if args.mtx:
        print(f"Loading matrix from {args.mtx} ...")
        A = read_matrix_market(args.mtx)
        n = A.shape[0]
        if args.rhs:
            b = np.load(args.rhs)
            if b.shape[0] != n:
                raise ValueError("RHS size mismatch")
        else:
            b = np.ones(n, dtype=np.float64)
    else:
        n = args.n
        if args.matrix_type == 'tri' or args.half_band == 1:
            print(f"Building tridiagonal matrix of size {n}")
            if args.return_diagonals:  # internal debug flag
                a_diag, b_diag, c_diag = build_tridiagonal_diags(n)
                A = None
            else:
                A = build_tridiagonal_sparse(n)
                a_diag, b_diag, c_diag = None, None, None
        else:
            print(f"Building banded matrix size {n} half_band {args.half_band}")
            A = build_banded_sparse(n, half_band=args.half_band)
            a_diag, b_diag, c_diag = None, None, None
        if args.rhs:
            b = np.load(args.rhs)
        else:
            b = np.ones(n, dtype=np.float64)
    # decide whether to use TDMA
    use_tdma = False
    extracted_diags = None
    if args.method.lower() == 'tdma':
        use_tdma = True
    elif args.auto_tdma:
        # auto-detect if A is tridiagonal (only possible if A exists)
        if 'A' in locals() and A is not None:
            try:
                lower_diag, main_diag, upper_diag = extract_tridiagonal_from_sparse(A)
                use_tdma = True
                extracted_diags = (lower_diag, main_diag, upper_diag)
            except ValueError:
                use_tdma = False
        else:
            # if we built diagonals earlier (return_diagonals path)
            if 'a_diag' in locals() and a_diag is not None:
                extracted_diags = (a_diag, b_diag, c_diag)
                use_tdma = True
    # If user built tri-diagonals but we only have diag arrays, fetch them
    if use_tdma:
        # get diagonals a, b, c from whatever source
        if extracted_diags is not None:
            a_arr, b_arr, c_arr = extracted_diags
        else:
            # maybe we have a_diag, etc.
            if 'a_diag' in locals() and a_diag is not None:
                a_arr, b_arr, c_arr = a_diag, b_diag, c_diag
            else:
                # if A exists as sparse, extract
                a_arr, b_arr, c_arr = extract_tridiagonal_from_sparse(A)
        # sanity checks
        if not (a_arr.dtype == b_arr.dtype == c_arr.dtype == np.float64):
            a_arr = a_arr.astype(np.float64)
            b_arr = b_arr.astype(np.float64)
            c_arr = c_arr.astype(np.float64)
        # call TDMA
        print(f"Solving with TDMA (n={b_arr.size}) ...")
        t0 = time.time()
        x = tdma_solve(a_arr, b_arr, c_arr, b)  # here `b` is RHS; name clash but clear
        meta = {'solver': 'tdma', 'time': time.time() - t0}
        # compute residual (without building full A)
        # build Ax manually for tridiagonal
        Ax = np.empty_like(b)
        Ax[0] = b_arr[0]*x[0] + c_arr[0]*x[1]
        Ax[1:-1] = a_arr[:-1]*x[:-2] + b_arr[1:-1]*x[1:-1] + c_arr[1:]*x[2:]
        Ax[-1] = a_arr[-1]*x[-2] + b_arr[-1]*x[-1]
        meta['residual_norm'] = float(np.linalg.norm(Ax - b))
        return x, meta
    # Otherwise fallback to SciPy solvers
    if 'A' not in locals() or A is None:
        # If we only have diagonals (rare branch), build sparse A now
        print("Converting diagonals to sparse matrix ...")
        A = sp.diags([a_diag, b_diag, c_diag], offsets=[-1, 0, 1], format='csr', dtype=np.float64)
    # call SciPy wrapper
    print(f"Solving with SciPy method='{args.method}' (use_ilu={args.use_ilu}) ...")
    x, meta = scipy_solve(A, b, method=args.method, tol=args.tol, maxiter=args.maxiter, use_ilu=args.use_ilu)
    return x, meta

# --- Command-line interface ---
def parse_args():
    parser = argparse.ArgumentParser(description="Expanded sparse solver supporting TDMA + SciPy")
    parser.add_argument('--mtx', type=str, default=None, help='Path to Matrix Market (.mtx) file for A')
    parser.add_argument('--rhs', type=str, default=None, help='Path to numpy .npy file for RHS b (optional)')
    parser.add_argument('--n', type=int, default=1000000, help='Size n for synthetic matrix (default 1e6)')
    parser.add_argument('--matrix_type', choices=['tri', 'band'], default='tri', help='Synthetic matrix type')
    parser.add_argument('--half_band', type=int, default=3, help='Half-bandwidth for banded synthetic matrix')
    parser.add_argument('--method', choices=['direct', 'cg', 'bicgstab', 'gmres', 'tdma'], default='tdma', help='Solver method (tdma chooses TDMA when available)')
    parser.add_argument('--use_ilu', action='store_true', help='Use ILU preconditioner for iterative SciPy solvers')
    parser.add_argument('--tol', type=float, default=1e-8, help='Tolerance for iterative solver')
    parser.add_argument('--maxiter', type=int, default=None, help='Max iterations for iterative solver')
    parser.add_argument('--auto_tdma', action='store_true', help='Auto-detect tridiagonal structure and use TDMA if possible')
    # internal flag: when building tridiagonal, optionally return diagonals instead of sparse A
    parser.add_argument('--return_diagonals', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()

# --- Main execution ---
def main():
    args = parse_args()
    print("Arguments:", args)
    x, meta = solve_system(args)
    print("=== SOLVE METADATA ===")
    for k, v in meta.items():
        print(f"{k:20s} : {v}")
    print("Solution sample (first 10 entries):", x[:10])
    # optionally save solution
    # if args.save_solution: np.save(args.save_solution, x)

if __name__ == "__main__":
    main()
