def compute_quick_correction(phi, Nx, Ny, dx, dy, ux_grid, uy_grid):
    """
    Compute the QUICK scheme correction term with Van Leer limiter.

    HAYASE'S QUICK SCHEME - DETAILED EXPLANATION:
    =============================================

    This implements a TVD-QUICK scheme using Hayase's deferred correction approach.
    The method combines the stability of upwind with the accuracy of QUICK.

    DEFERRED CORRECTION METHOD:
    --------------------------
    Instead of solving the full QUICK system directly, we split it:

        A_upwind * φ^(n+1) = b + CORRECTION(φ^n)

    where:
    - A_upwind: Stable first-order upwind matrix (implicit)
    - CORRECTION: Higher-order terms based on previous solution (explicit)

    This is ITERATIVE:
    1. Solve with upwind → get φ^0
    2. Compute correction based on φ^0
    3. Solve again with corrected RHS → get φ^1
    4. Repeat until φ converges


    QUICK FACE VALUE INTERPOLATION:
    -------------------------------
    For a face between cells U (upwind) and D (downwind):

    Standard upwind (1st order):
        φ_face = φ_U

    QUICK with limiting (2nd-3rd order):
        φ_face = φ_U + 0.5 × ψ(r) × (φ_D - φ_UU) / 2

    Breaking this down:

    1. BASE: φ_U (upwind value - stable)

    2. CORRECTION: 0.5 × ψ(r) × (φ_D - φ_UU) / 2
       - (φ_D - φ_UU): Measures gradient across 3 cells
       - 0.5: Geometric factor (face is halfway between cells)
       - ψ(r): Van Leer limiter (0 to 1)
       - Division by 2: Normalizes the three-point stencil

    3. When ψ(r) = 0: φ_face = φ_U (reverts to upwind)
    4. When ψ(r) = 1: φ_face = φ_U + correction (full QUICK)


    GRADIENT RATIO:
    --------------
    The key to limiting is computing r at each face:

        r = (φ_U - φ_UU) / (φ_D - φ_U)

    Visualization for flow W → P → E:

        [φ_W]  →  [φ_P]  →  [φ_E]
          ↑         ↑         ↑
         UU        U         D

    At east face:
    - Upwind: φ_U = φ_P (current cell)
    - Downwind: φ_D = φ_E (east cell)
    - Far upwind: φ_UU = φ_W (west cell)

    r measures if the solution is smooth:
    - r ≈ 1: Uniform gradient → safe to use QUICK
    - r < 0: Gradient changes sign → use upwind only
    - r >> 1: Steep upwind gradient → limit QUICK


    FLUX CORRECTION:
    ---------------
    The correction to the RHS is:

        Correction = Flux_QUICK - Flux_upwind

    For the east face:
        Flux_upwind = u_e × φ_U
        Flux_QUICK = u_e × (φ_U + 0.5×ψ×(φ_D - φ_UU)/2)

        Correction = u_e × 0.5 × ψ(r) × (φ_D - φ_UU) / 2

    This correction is added to the RHS for each interior cell based on
    all four faces (east, west, north, south).


    WHY THIS WORKS:
    --------------
    1import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os

# ==================== PROBLEM PARAMETERS ====================
Lx, Ly = 1.0, 1.0  # Domain dimensions
gamma = 0.0  # Diffusion coefficient (will be changed for different cases)
meshes = [(80, 80), (160, 160), (320, 320)]  # Three mesh sizes for convergence study

# Boundary conditions (Dirichlet on all sides)
phi_left = 100.0    # x = 0
phi_right = 0.0     # x = Lx
phi_bottom = 0.0    # y = 0
phi_top = 100.0     # y = Ly


# ==================== GRID GENERATION ====================
def build_grid(Nx, Ny):
    """
    '''
    Build uniform grid with cell centers.

    Parameters:
    -----------
    Nx, Ny : int
        Number of cells in x and y directions

    Returns:
    --------
    x_c, y_c : ndarray
        Cell center coordinates
    dx, dy : float
        Cell spacing in x and y directions
    '''
    dx = Lx / Nx  # Uniform spacing in x
    dy = Ly / Ny  # Uniform spacing in y

    # Cell centers
    x_c = np.linspace(dx / 2, Lx - dx / 2, Nx)
    y_c = np.linspace(dy / 2, Ly - dy / 2, Ny)

    return x_c, y_c, dx, dy


# ==================== VELOCITY FIELDS ====================
def velocity_constant(ux_val, uy_val):
    """Create constant velocity field: u = (ux_val, uy_val)"""

    def ux_func(x, y):
        return ux_val

    def uy_func(x, y):
        return uy_val

    return ux_func, uy_func


def velocity_rotational():
    """Create rotational velocity field: u = (-r*sin(θ), r*cos(θ))"""

    def ux_func(x, y):
        r = np.sqrt((x - Lx / 2) ** 2 + (y - Ly / 2) ** 2)
        theta = np.arctan2(y - Ly / 2, x - Lx / 2)
        return -r * np.sin(theta)

    def uy_func(x, y):
        r = np.sqrt((x - Lx / 2) ** 2 + (y - Ly / 2) ** 2)
        theta = np.arctan2(y - Ly / 2, x - Lx / 2)
        return r * np.cos(theta)

    return ux_func, uy_func


# ==================== INDEX MAPPING ====================
def idx(i, j, Nx):
    """Map 2D cell indices (i,j) to 1D matrix index (row-major ordering)"""
    return j * Nx + i


# ==================== SLOPE LIMITERS ====================
def van_leer_limiter(r):
    """
    Van Leer slope limiter for TVD schemes.

    DETAILED EXPLANATION:
    ====================

    The Van Leer limiter is used to prevent oscillations in high-order schemes
    while maintaining accuracy in smooth regions.

    Formula: ψ(r) = (r + |r|) / (1 + |r|)

    where r is the gradient ratio:
        r = (φ_U - φ_UU) / (φ_D - φ_U)

    Components:
    -----------
    φ_UU : Far upwind cell value
    φ_U  : Upwind cell value (where flow is coming FROM)
    φ_D  : Downwind cell value (where flow is going TO)

    Interpretation of r:
    -------------------
    - Numerator (φ_U - φ_UU):   Gradient on the upwind side
    - Denominator (φ_D - φ_U):  Gradient on the downwind side

    r > 0: Gradients have same sign (smooth, monotonic region)
    r = 1: Gradients are equal (perfectly smooth)
    r < 0: Gradients have opposite signs (local extremum/inflection)
    r → ∞: Large upwind gradient compared to downwind

    Limiter Behavior:
    ----------------
    ψ(r ≤ 0) = 0     → REVERT TO UPWIND (1st order, stable)
    ψ(r = 1) = 1     → FULL HIGHER-ORDER reconstruction
    ψ(r → ∞) → 1     → FULL HIGHER-ORDER reconstruction
    0 < ψ(r) < 1     → BLEND between upwind and higher-order

    Properties:
    ----------
    1. TVD (Total Variation Diminishing): Prevents new extrema
    2. Smooth and differentiable everywhere
    3. Symmetric: ψ(1/r) / r = ψ(r)
    4. Second-order accurate in smooth regions

    Visual Example:
    --------------
    Smooth region: φ = [10, 20, 30, 40]
        r = (20-10)/(30-20) = 1.0
        ψ(1) = 1.0 → Use full QUICK

    Local peak: φ = [10, 30, 25, 35]
        r = (30-10)/(25-30) = 20/(-5) = -4.0
        ψ(-4) = 0 → Fall back to upwind (prevents overshoot)

    Parameters:
    -----------
    r : float or ndarray
        Ratio of consecutive gradients

    Returns:
    --------
    limiter : float or ndarray
        Limited slope (between 0 and 1)
    """
    # Van Leer limiter: φ(r) = (r + |r|) / (1 + |r|)
    # Returns 0 if r ≤ 0, otherwise smooth transition to 2*r/(1+r)
    return (r + np.abs(r)) / (1.0 + np.abs(r))


# ==================== MATRIX ASSEMBLY ====================
def assemble_system(Nx, Ny, ux_func, uy_func, scheme='central'):
    """
    Assemble the sparse matrix A and RHS vector b for the equation:
    ∇·(u*φ) = ∇·(Γ∇φ)

    Parameters:
    -----------
    Nx, Ny : int
        Number of cells in x and y directions
    ux_func, uy_func : callable
        Functions that return velocity components given (x, y)
    scheme : str
        Advection scheme: 'central', 'upwind', or 'quick'

    Returns:
    --------
    A : sparse matrix
        Coefficient matrix
    b : ndarray
        Right-hand side vector
    x_c, y_c : ndarray
        Cell center coordinates
    """
    x_c, y_c, dx, dy = build_grid(Nx, Ny)
    N = Nx * Ny

    # Lists to build sparse matrix in COO format
    rows, cols, data = [], [], []
    b = np.zeros(N)

    # Create 2D meshgrid for velocity evaluation
    X, Y = np.meshgrid(x_c, y_c)
    ux_grid = ux_func(X, Y)  # Velocity at all cell centers
    uy_grid = uy_func(X, Y)

    # Loop through all cells
    for j in range(Ny):
        for i in range(Nx):
            p = idx(i, j, Nx)  # Current cell index
            aP = 0.0  # Coefficient for center cell P

            # Check if on boundary
            on_left = (i == 0)
            on_right = (i == Nx - 1)
            on_bottom = (j == 0)
            on_top = (j == Ny - 1)

            # ==================== BOUNDARY CONDITIONS ====================
            if on_left:
                # Dirichlet BC at x = 0
                rows.append(p)
                cols.append(p)
                data.append(1.0)
                b[p] = phi_left
                continue
            elif on_right:
                # Dirichlet BC at x = Lx
                rows.append(p)
                cols.append(p)
                data.append(1.0)
                b[p] = phi_right
                continue
            elif on_bottom:
                # Dirichlet BC at y = 0
                rows.append(p)
                cols.append(p)
                data.append(1.0)
                b[p] = phi_bottom
                continue
            elif on_top:
                # Dirichlet BC at y = Ly
                rows.append(p)
                cols.append(p)
                data.append(1.0)
                b[p] = phi_top
                continue

            # ==================== INTERIOR CELLS ====================
            # Get velocity at current cell
            ux_P = ux_grid[j, i]
            uy_P = uy_grid[j, i]

            # --- DIFFUSION TERMS (always second-order central difference) ---
            # East neighbor (i+1, j)
            aE_diff = gamma / dx ** 2
            rows.append(p)
            cols.append(idx(i + 1, j, Nx))
            data.append(-aE_diff)
            aP += aE_diff

            # West neighbor (i-1, j)
            aW_diff = gamma / dx ** 2
            rows.append(p)
            cols.append(idx(i - 1, j, Nx))
            data.append(-aW_diff)
            aP += aW_diff

            # North neighbor (i, j+1)
            aN_diff = gamma / dy ** 2
            rows.append(p)
            cols.append(idx(i, j + 1, Nx))
            data.append(-aN_diff)
            aP += aN_diff

            # South neighbor (i, j-1)
            aS_diff = gamma / dy ** 2
            rows.append(p)
            cols.append(idx(i, j - 1, Nx))
            data.append(-aS_diff)
            aP += aS_diff

            # --- ADVECTION TERMS ---
            # Interpolate velocities to cell faces
            ux_e = (ux_grid[j, i] + ux_grid[j, i + 1]) / 2.0  # East face
            ux_w = (ux_grid[j, i - 1] + ux_grid[j, i]) / 2.0  # West face
            uy_n = (uy_grid[j, i] + uy_grid[j + 1, i]) / 2.0  # North face
            uy_s = (uy_grid[j - 1, i] + uy_grid[j, i]) / 2.0  # South face

            if scheme == 'central':
                # CENTRAL DIFFERENCE SCHEME
                # ∂(u*φ)/∂x ≈ (u_e*φ_e - u_w*φ_w)/dx with φ_e = (φ_E + φ_P)/2

                # x-direction advection
                # East face contribution
                rows.append(p)
                cols.append(idx(i + 1, j, Nx))  # φ_E coefficient
                data.append(ux_e / (2.0 * dx))
                aP += ux_e / (2.0 * dx)  # φ_P coefficient from east face

                # West face contribution
                rows.append(p)
                cols.append(idx(i - 1, j, Nx))  # φ_W coefficient
                data.append(-ux_w / (2.0 * dx))
                aP -= ux_w / (2.0 * dx)  # φ_P coefficient from west face

                # y-direction advection
                # North face contribution
                rows.append(p)
                cols.append(idx(i, j + 1, Nx))  # φ_N coefficient
                data.append(uy_n / (2.0 * dy))
                aP += uy_n / (2.0 * dy)  # φ_P coefficient from north face

                # South face contribution
                rows.append(p)
                cols.append(idx(i, j - 1, Nx))  # φ_S coefficient
                data.append(-uy_s / (2.0 * dy))
                aP -= uy_s / (2.0 * dy)  # φ_P coefficient from south face

            elif scheme == 'upwind':
                # FIRST-ORDER UPWIND SCHEME
                # Choose upwind direction based on velocity sign

                # x-direction advection
                # East face: flux = ux_e * φ_upwind
                if ux_e > 0:
                    # Flow from P to E, use φ_P
                    aP -= ux_e / dx
                else:
                    # Flow from E to P, use φ_E
                    rows.append(p)
                    cols.append(idx(i + 1, j, Nx))
                    data.append(-ux_e / dx)

                # West face: flux = ux_w * φ_upwind
                if ux_w > 0:
                    # Flow from W to P, use φ_W
                    rows.append(p)
                    cols.append(idx(i - 1, j, Nx))
                    data.append(ux_w / dx)
                else:
                    # Flow from P to W, use φ_P
                    aP += ux_w / dx

                # y-direction advection
                # North face: flux = uy_n * φ_upwind
                if uy_n > 0:
                    # Flow from P to N, use φ_P
                    aP -= uy_n / dy
                else:
                    # Flow from N to P, use φ_N
                    rows.append(p)
                    cols.append(idx(i, j + 1, Nx))
                    data.append(-uy_n / dy)

                # South face: flux = uy_s * φ_upwind
                if uy_s > 0:
                    # Flow from S to P, use φ_S
                    rows.append(p)
                    cols.append(idx(i, j - 1, Nx))
                    data.append(uy_s / dy)
                else:
                    # Flow from P to S, use φ_P
                    aP += uy_s / dy

            elif scheme == 'quick':
                # HAYASE'S QUICK SCHEME with Van Leer limiter
                #
                # This uses a deferred correction approach:
                # A_upwind * φ^(n+1) = b + CORRECTION(φ^n)
                #
                # The correction term uses QUICK interpolation with TVD limiting
                #
                # For the initial matrix assembly, we use UPWIND as base scheme
                # The QUICK correction will be applied iteratively

                # Base scheme: FIRST-ORDER UPWIND (same as upwind scheme)

                # East face
                if ux_e > 0:
                    aP -= ux_e / dx
                else:
                    rows.append(p)
                    cols.append(idx(i + 1, j, Nx))
                    data.append(-ux_e / dx)

                # West face
                if ux_w > 0:
                    rows.append(p)
                    cols.append(idx(i - 1, j, Nx))
                    data.append(ux_w / dx)
                else:
                    aP += ux_w / dx

                # North face
                if uy_n > 0:
                    aP -= uy_n / dy
                else:
                    rows.append(p)
                    cols.append(idx(i, j + 1, Nx))
                    data.append(-uy_n / dy)

                # South face
                if uy_s > 0:
                    rows.append(p)
                    cols.append(idx(i, j - 1, Nx))
                    data.append(uy_s / dy)
                else:
                    aP += uy_s / dy
                # QUICK SCHEME with Van Leer slope limiter
                # Uses piecewise linear reconstruction: φ_face = φ_upwind + 0.5*ψ(r)*∇φ*Δx
                # where ψ(r) is the Van Leer limiter and r is the gradient ratio

                # This requires an iterative approach since we need φ values
                # For direct matrix solution, we'll use a deferred correction method:
                # 1. Base scheme: first-order upwind (implicit in matrix)
                # 2. Higher-order terms: treated explicitly (requires iteration)

                # For simplicity in this assignment, implement QUICK as:
                # φ_face = weighted average of upwind, downwind, and far-upwind values

                # x-direction advection
                # East face
                if ux_e > 0:
                    # Flow P→E, reconstruction point is P
                    if i >= 1:
                        # Standard QUICK: φ_e = (6φ_P + 3φ_E - φ_W)/8
                        # With limiting: blend between upwind (φ_P) and QUICK
                        # Matrix form: aW*φ_W + aP*φ_P + aE*φ_E
                        rows.append(p)
                        cols.append(idx(i - 1, j, Nx))
                        data.append(ux_e / (8.0 * dx))  # -1/8 contribution from φ_W

                        aP -= ux_e * 6.0 / (8.0 * dx)  # 6/8 from φ_P

                        rows.append(p)
                        cols.append(idx(i + 1, j, Nx))
                        data.append(-ux_e * 3.0 / (8.0 * dx))  # 3/8 from φ_E
                    else:
                        # Near boundary, fall back to upwind
                        aP -= ux_e / dx
                else:
                    # Flow E→P, reconstruction point is E
                    if i <= Nx - 2:
                        # QUICK: φ_e = (6φ_E + 3φ_P - φ_EE)/8
                        rows.append(p)
                        cols.append(idx(i + 1, j, Nx))
                        data.append(-ux_e * 6.0 / (8.0 * dx))  # 6/8 from φ_E

                        aP += ux_e * 3.0 / (8.0 * dx)  # 3/8 from φ_P

                        if i <= Nx - 3:
                            rows.append(p)
                            cols.append(idx(i + 2, j, Nx))
                            data.append(ux_e / (8.0 * dx))  # -1/8 from φ_EE
                        else:
                            # Near boundary, adjust to second-order
                            rows.append(p)
                            cols.append(idx(i + 1, j, Nx))
                            data.append(-ux_e * 3.0 / (8.0 * dx))
                    else:
                        # Fall back to upwind
                        rows.append(p)
                        cols.append(idx(i + 1, j, Nx))
                        data.append(-ux_e / dx)

                # West face
                if ux_w > 0:
                    # Flow W→P, reconstruction point is W
                    if i >= 2:
                        # QUICK: φ_w = (6φ_W + 3φ_P - φ_WW)/8
                        rows.append(p)
                        cols.append(idx(i - 1, j, Nx))
                        data.append(ux_w * 6.0 / (8.0 * dx))  # 6/8 from φ_W

                        aP -= ux_w * 3.0 / (8.0 * dx)  # 3/8 from φ_P

                        rows.append(p)
                        cols.append(idx(i - 2, j, Nx))
                        data.append(-ux_w / (8.0 * dx))  # -1/8 from φ_WW
                    else:
                        # Fall back to upwind
                        rows.append(p)
                        cols.append(idx(i - 1, j, Nx))
                        data.append(ux_w / dx)
                else:
                    # Flow P→W, reconstruction point is P
                    if i >= 1:
                        # QUICK: φ_w = (6φ_P + 3φ_W - φ_E)/8
                        aP += ux_w * 6.0 / (8.0 * dx)  # 6/8 from φ_P

                        rows.append(p)
                        cols.append(idx(i - 1, j, Nx))
                        data.append(-ux_w * 3.0 / (8.0 * dx))  # 3/8 from φ_W

                        if i <= Nx - 2:
                            rows.append(p)
                            cols.append(idx(i + 1, j, Nx))
                            data.append(ux_w / (8.0 * dx))  # -1/8 from φ_E
                        else:
                            # Adjust near boundary
                            rows.append(p)
                            cols.append(idx(i - 1, j, Nx))
                            data.append(-ux_w * 3.0 / (8.0 * dx))
                    else:
                        aP += ux_w / dx

                # y-direction advection
                # North face
                if uy_n > 0:
                    # Flow P→N, reconstruction point is P
                    if j >= 1:
                        # QUICK: φ_n = (6φ_P + 3φ_N - φ_S)/8
                        rows.append(p)
                        cols.append(idx(i, j - 1, Nx))
                        data.append(uy_n / (8.0 * dy))  # -1/8 from φ_S

                        aP -= uy_n * 6.0 / (8.0 * dy)  # 6/8 from φ_P

                        rows.append(p)
                        cols.append(idx(i, j + 1, Nx))
                        data.append(-uy_n * 3.0 / (8.0 * dy))  # 3/8 from φ_N
                    else:
                        # Fall back to upwind
                        aP -= uy_n / dy
                else:
                    # Flow N→P, reconstruction point is N
                    if j <= Ny - 2:
                        # QUICK: φ_n = (6φ_N + 3φ_P - φ_NN)/8
                        rows.append(p)
                        cols.append(idx(i, j + 1, Nx))
                        data.append(-uy_n * 6.0 / (8.0 * dy))  # 6/8 from φ_N

                        aP += uy_n * 3.0 / (8.0 * dy)  # 3/8 from φ_P

                        if j <= Ny - 3:
                            rows.append(p)
                            cols.append(idx(i, j + 2, Nx))
                            data.append(uy_n / (8.0 * dy))  # -1/8 from φ_NN
                        else:
                            # Adjust near boundary
                            rows.append(p)
                            cols.append(idx(i, j + 1, Nx))
                            data.append(-uy_n * 3.0 / (8.0 * dy))
                    else:
                        # Fall back to upwind
                        rows.append(p)
                        cols.append(idx(i, j + 1, Nx))
                        data.append(-uy_n / dy)

                # South face
                if uy_s > 0:
                    # Flow S→P, reconstruction point is S
                    if j >= 2:
                        # QUICK: φ_s = (6φ_S + 3φ_P - φ_SS)/8
                        rows.append(p)
                        cols.append(idx(i, j - 1, Nx))
                        data.append(uy_s * 6.0 / (8.0 * dy))  # 6/8 from φ_S

                        aP -= uy_s * 3.0 / (8.0 * dy)  # 3/8 from φ_P

                        rows.append(p)
                        cols.append(idx(i, j - 2, Nx))
                        data.append(-uy_s / (8.0 * dy))  # -1/8 from φ_SS
                    else:
                        # Fall back to upwind
                        rows.append(p)
                        cols.append(idx(i, j - 1, Nx))
                        data.append(uy_s / dy)
                else:
                    # Flow P→S, reconstruction point is P
                    if j >= 1:
                        # QUICK: φ_s = (6φ_P + 3φ_S - φ_N)/8
                        aP += uy_s * 6.0 / (8.0 * dy)  # 6/8 from φ_P

                        rows.append(p)
                        cols.append(idx(i, j - 1, Nx))
                        data.append(-uy_s * 3.0 / (8.0 * dy))  # 3/8 from φ_S

                        if j <= Ny - 2:
                            rows.append(p)
                            cols.append(idx(i, j + 1, Nx))
                            data.append(uy_s / (8.0 * dy))  # -1/8 from φ_N
                        else:
                            # Adjust near boundary
                            rows.append(p)
                            cols.append(idx(i, j - 1, Nx))
                            data.append(-uy_s * 3.0 / (8.0 * dy))
                    else:
                        aP += uy_s / dy

            # Add diagonal coefficient
            rows.append(p)
            cols.append(p)
            data.append(aP)

    # Create sparse matrix in CSR format
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    return A, b, x_c, y_c


# ==================== SOLVER ====================
def solve_system(A, b):
    """Solve the linear system A*φ = b using sparse direct solver"""
    phi_vec = spla.spsolve(A, b)
    return phi_vec


def compute_quick_correction(phi, Nx, Ny, dx, dy, ux_grid, uy_grid):
    """
    Compute the QUICK scheme correction term with Van Leer limiter.

    This implements Hayase's deferred correction approach for QUICK:
    The correction represents the difference between QUICK and upwind fluxes.

    Parameters:
    -----------
    phi : ndarray (Ny, Nx)
        Current solution field
    Nx, Ny : int
        Grid dimensions
    dx, dy : float
        Cell spacing
    ux_grid, uy_grid : ndarray
        Velocity fields at cell centers

    Returns:
    --------
    correction : ndarray (Ny*Nx,)
        Correction term to add to RHS
    """
    correction = np.zeros(Nx * Ny)

    for j in range(Ny):
        for i in range(Nx):
            # Skip boundary cells
            if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                continue

            p = idx(i, j, Nx)
            corr = 0.0

            # Interpolate velocities to faces
            ux_e = (ux_grid[j, i] + ux_grid[j, i + 1]) / 2.0
            ux_w = (ux_grid[j, i - 1] + ux_grid[j, i]) / 2.0
            uy_n = (uy_grid[j, i] + uy_grid[j + 1, i]) / 2.0
            uy_s = (uy_grid[j - 1, i] + uy_grid[j, i]) / 2.0

            # ========== EAST FACE ==========
            if ux_e > 0:
                # Flow: W → P → E (upwind is P)
                if i >= 1 and i < Nx - 1:
                    phi_UU = phi[j, i - 1]  # West cell (far upwind)
                    phi_U = phi[j, i]  # Current cell (upwind)
                    phi_D = phi[j, i + 1]  # East cell (downwind)

                    # Compute gradient ratio r = (φ_U - φ_UU) / (φ_D - φ_U)
                    denom = phi_D - phi_U
                    if abs(denom) > 1e-10:
                        r = (phi_U - phi_UU) / denom

                        # Van Leer limiter: ψ(r) = (r + |r|) / (1 + |r|)
                        psi = (r + abs(r)) / (1.0 + abs(r))

                        # QUICK correction for east face
                        # φ_face_upwind = φ_U (first order)
                        # φ_face_QUICK = φ_U + 0.5*ψ*(φ_D - φ_UU)/2
                        # Correction = ux_e * (φ_face_QUICK - φ_face_upwind) / dx
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr -= ux_e * quick_correction / dx
            else:
                # Flow: E → P → W (upwind is E)
                if i >= 0 and i < Nx - 2:
                    phi_U = phi[j, i + 1]  # East cell (upwind)
                    phi_D = phi[j, i]  # Current cell (downwind)
                    if i < Nx - 2:
                        phi_UU = phi[j, i + 2]  # Far east (far upwind)
                    else:
                        phi_UU = phi_U  # Use upwind if no far neighbor

                    denom = phi_D - phi_U
                    if abs(denom) > 1e-10:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + abs(r))
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr -= ux_e * quick_correction / dx

            # ========== WEST FACE ==========
            if ux_w > 0:
                # Flow: WW → W → P (upwind is W)
                if i >= 2:
                    phi_UU = phi[j, i - 2]  # Far west (far upwind)
                    phi_U = phi[j, i - 1]  # West cell (upwind)
                    phi_D = phi[j, i]  # Current cell (downwind)

                    denom = phi_D - phi_U
                    if abs(denom) > 1e-10:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + abs(r))
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr += ux_w * quick_correction / dx
            else:
                # Flow: P → W → WW (upwind is P)
                if i >= 1:
                    phi_U = phi[j, i]  # Current cell (upwind)
                    phi_D = phi[j, i - 1]  # West cell (downwind)
                    if i >= 1:
                        phi_UU = phi[j, i + 1]  # East (far upwind)
                    else:
                        phi_UU = phi_U

                    denom = phi_D - phi_U
                    if abs(denom) > 1e-10:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + abs(r))
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr += ux_w * quick_correction / dx

            # ========== NORTH FACE ==========
            if uy_n > 0:
                # Flow: S → P → N (upwind is P)
                if j >= 1 and j < Ny - 1:
                    phi_UU = phi[j - 1, i]  # South cell (far upwind)
                    phi_U = phi[j, i]  # Current cell (upwind)
                    phi_D = phi[j + 1, i]  # North cell (downwind)

                    denom = phi_D - phi_U
                    if abs(denom) > 1e-10:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + abs(r))
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr -= uy_n * quick_correction / dy
            else:
                # Flow: N → P → S (upwind is N)
                if j >= 0 and j < Ny - 2:
                    phi_U = phi[j + 1, i]  # North cell (upwind)
                    phi_D = phi[j, i]  # Current cell (downwind)
                    if j < Ny - 2:
                        phi_UU = phi[j + 2, i]  # Far north (far upwind)
                    else:
                        phi_UU = phi_U

                    denom = phi_D - phi_U
                    if abs(denom) > 1e-10:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + abs(r))
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr -= uy_n * quick_correction / dy

            # ========== SOUTH FACE ==========
            if uy_s > 0:
                # Flow: SS → S → P (upwind is S)
                if j >= 2:
                    phi_UU = phi[j - 2, i]  # Far south (far upwind)
                    phi_U = phi[j - 1, i]  # South cell (upwind)
                    phi_D = phi[j, i]  # Current cell (downwind)

                    denom = phi_D - phi_U
                    if abs(denom) > 1e-10:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + abs(r))
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr += uy_s * quick_correction / dy
            else:
                # Flow: P → S → SS (upwind is P)
                if j >= 1:
                    phi_U = phi[j, i]  # Current cell (upwind)
                    phi_D = phi[j - 1, i]  # South cell (downwind)
                    if j >= 1:
                        phi_UU = phi[j + 1, i]  # North (far upwind)
                    else:
                        phi_UU = phi_U

                    denom = phi_D - phi_U
                    if abs(denom) > 1e-10:
                        r = (phi_U - phi_UU) / denom
                        psi = (r + abs(r)) / (1.0 + abs(r))
                        quick_correction = 0.5 * psi * (phi_D - phi_UU) / 2.0
                        corr += uy_s * quick_correction / dy

            correction[p] = corr

    return correction


def solve_with_quick_iterations(Nx, Ny, ux_func, uy_func, max_iter=10, tol=1e-6):
    """
    Solve using QUICK scheme with iterative deferred correction.

    Algorithm:
    1. Assemble upwind matrix A and RHS b
    2. Solve A*φ = b (initial solution)
    3. Compute QUICK correction based on current φ
    4. Update RHS: b_new = b + correction
    5. Solve again: A*φ_new = b_new
    6. Repeat until converged

    Parameters:
    -----------
    Nx, Ny : int
        Grid dimensions
    ux_func, uy_func : callable
        Velocity functions
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance

    Returns:
    --------
    xc, yc : ndarray
        Cell centers
    phi : ndarray
        Converged solution
    n_iter : int
        Number of iterations taken
    """
    # Build grid and velocity field
    xc, yc, dx, dy = build_grid(Nx, Ny)
    X, Y = np.meshgrid(xc, yc)
    ux_grid = ux_func(X, Y)
    uy_grid = uy_func(X, Y)

    # Assemble base upwind matrix (this doesn't change)
    A, b_base, _, _ = assemble_system(Nx, Ny, ux_func, uy_func, 'quick')

    # Initial solution with upwind
    phi_vec = solve_system(A, b_base)
    phi = phi_to_grid(phi_vec, Nx, Ny)

    # Iterative correction loop
    for iteration in range(max_iter):
        # Compute QUICK correction based on current solution
        correction = compute_quick_correction(phi, Nx, Ny, dx, dy, ux_grid, uy_grid)

        # Update RHS with correction
        b_corrected = b_base + correction

        # Solve with corrected RHS
        phi_vec_new = solve_system(A, b_corrected)
        phi_new = phi_to_grid(phi_vec_new, Nx, Ny)

        # Check convergence
        change = np.max(np.abs(phi_new - phi))

        if iteration == 0 or iteration % 2 == 0:
            print(f"    QUICK iteration {iteration + 1}: max change = {change:.6e}")

        if change < tol:
            print(f"    QUICK converged in {iteration + 1} iterations")
            return xc, yc, phi_new, iteration + 1

        phi = phi_new
        phi_vec = phi_vec_new

    print(f"    QUICK reached max iterations ({max_iter})")
    return xc, yc, phi, max_iter


def phi_to_grid(phi_vec, Nx, Ny):
    """Convert 1D solution vector to 2D grid"""
    return phi_vec.reshape((Ny, Nx))


# ==================== PLOTTING ====================
def plot_phi(xc, yc, phi, title, fname, ux_func=None, uy_func=None):
    """
    Create and save contour plot of solution with optional velocity vectors.

    Parameters:
    -----------
    xc, yc : ndarray
        Cell center coordinates
    phi : ndarray
        Solution field
    title : str
        Plot title
    fname : str
        Filename to save
    ux_func, uy_func : callable, optional
        Velocity functions for plotting streamlines/vectors
    """
    X, Y = np.meshgrid(xc, yc)
    fig, ax = plt.subplots(figsize=(8, 6))

    levels = np.linspace(phi.min(), phi.max(), 21)
    cs = plt.contourf(X, Y, phi, levels=levels, cmap='jet')
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('φ', fontsize=11)

    # Add velocity vectors if provided
    if ux_func is not None and uy_func is not None:
        # Evaluate velocity field
        ux_grid = ux_func(X, Y)
        uy_grid = uy_func(X, Y)

        # Subsample for clearer visualization (every n-th point)
        skip = max(len(xc) // 15, 1)  # Show ~15 vectors per direction

        # Plot velocity vectors (quiver plot)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  ux_grid[::skip, ::skip], uy_grid[::skip, ::skip],
                  color='black', alpha=0.6, scale=50, width=0.003,
                  headwidth=4, headlength=5)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_velocity_field(xc, yc, ux_func, uy_func, title, fname):
    """
    Create a plot showing just the velocity field (like Figure 1 in assignment).

    Parameters:
    -----------
    xc, yc : ndarray
        Cell center coordinates
    ux_func, uy_func : callable
        Velocity functions
    title : str
        Plot title
    fname : str
        Filename to save
    """
    X, Y = np.meshgrid(xc, yc)
    ux_grid = ux_func(X, Y)
    uy_grid = uy_func(X, Y)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot velocity vectors
    skip = max(len(xc) // 12, 1)  # Show ~12 vectors per direction
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              ux_grid[::skip, ::skip], uy_grid[::skip, ::skip],
              color='black', alpha=0.8, scale=40, width=0.004,
              headwidth=4, headlength=5)

    # Add boundary condition labels
    ax.text(-0.05, 0.5, r'$\phi_b = 100


def plot_diagonal_comparison(results_dict, save_dir, case_name, gamma_val):
    """
    Create a comparison plot along the diagonal like Figure 5.15.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys like '80_central', '80_upwind', etc.
        Values are tuples of (xc, yc, phi)
    save_dir : str
        Directory to save the plot
    case_name : str
        Name for the plot file
    gamma_val : float
        Value of gamma for the title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define line styles and colors
    styles = {
        'central': {'linestyle': ':', 'linewidth': 2, 'color': 'black'},
        'upwind': {'linestyle': '-', 'linewidth': 2, 'color': 'black'}
    }

    resolutions = [80, 160, 320]

    for scheme in ['upwind', 'central']:
        for res in resolutions:
            key = f'{res}_{scheme}'
            if key not in results_dict:
                continue

            xc, yc, phi = results_dict[key]

            # Extract values along diagonal (x = y)
            # For each cell center, find the closest point to the diagonal
            nx, ny = len(xc), len(yc)
            X, Y = np.meshgrid(xc, yc)

            # Calculate distance along diagonal: d = sqrt(x^2 + y^2) for x=y line
            # For square domain, diagonal distance = sqrt(2) * x
            diagonal_distance = []
            phi_diagonal = []

            # Extract values close to the diagonal (within tolerance)
            tolerance = max(Lx / nx, Ly / ny) * 1.5  # Within 1.5 cell widths

            for j in range(ny):
                for i in range(nx):
                    x_val = X[j, i]
                    y_val = Y[j, i]

                    # Check if point is close to diagonal (x ≈ y)
                    if abs(x_val - y_val) < tolerance:
                        # Distance along diagonal from origin
                        dist = np.sqrt(x_val ** 2 + y_val ** 2)
                        diagonal_distance.append(dist)
                        phi_diagonal.append(phi[j, i])

            # Sort by distance
            sorted_indices = np.argsort(diagonal_distance)
            diagonal_distance = np.array(diagonal_distance)[sorted_indices]
            phi_diagonal = np.array(phi_diagonal)[sorted_indices]

            # Plot
            if scheme == 'upwind':
                if res == 80:
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='--', linewidth=2.5, color='black', label=label)
                elif res == 160:
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='-', linewidth=2, color='black', label=label)
                else:  # 320
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='-', linewidth=1.5, color='gray', label=label)
            else:  # central
                if res == 160:
                    label = f'Central {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle=':', linewidth=2.5, color='black', label=label)

    ax.set_xlabel('Distance along diagonal X = Y (m)', fontsize=12)
    ax.set_ylabel('φ', fontsize=12)
    ax.set_title(f'Solution along diagonal (Γ={gamma_val})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim(0, np.sqrt(2) * Lx)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'{case_name}_diagonal_comparison.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Diagonal comparison plot saved to: {fname}")


# ==================== CONVERGENCE ANALYSIS ====================
def compute_error_norm(phi_coarse, x_c, y_c, phi_ref, x_ref, y_ref):
    """
    Compute L2 error norm between coarse solution and reference solution.
    Uses bilinear interpolation to compare at coarse grid points.
    """
    # Create interpolator for reference solution
    interp_func = RegularGridInterpolator(
        (y_ref, x_ref),
        phi_ref,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Create meshgrid for coarse solution
    Yc, Xc = np.meshgrid(y_c, x_c, indexing='ij')
    points = np.column_stack([Yc.ravel(), Xc.ravel()])

    # Interpolate reference solution to coarse grid points
    phi_ref_on_coarse = interp_func(points).reshape(phi_coarse.shape)

    # Compute L2 error norm
    diff = phi_ref_on_coarse - phi_coarse
    error = np.sqrt(np.mean(diff ** 2))

    return error


# ==================== RUN SINGLE CASE ====================
def run_case(Nx, Ny, ux_func, uy_func, scheme, save_dir, case_name):
    """Run a single simulation case"""
    print(f"\nRunning {case_name}: {Nx}×{Ny} grid, {scheme} scheme...")
    os.makedirs(save_dir, exist_ok=True)

    if scheme == 'quick':
        # Use iterative QUICK solver
        xc, yc, phi, n_iter = solve_with_quick_iterations(Nx, Ny, ux_func, uy_func)

        # Compute residual manually
        A, b, _, _ = assemble_system(Nx, Ny, ux_func, uy_func, 'quick')
        X, Y = np.meshgrid(xc, yc)
        ux_grid = ux_func(X, Y)
        uy_grid = uy_func(X, Y)
        dx = Lx / Nx
        dy = Ly / Ny
        correction = compute_quick_correction(phi, Nx, Ny, dx, dy, ux_grid, uy_grid)
        b_corrected = b + correction
        phi_vec = phi.ravel()
        residual = np.linalg.norm(A @ phi_vec - b_corrected)
    else:
        # Standard assembly and solve
        A, b, xc, yc = assemble_system(Nx, Ny, ux_func, uy_func, scheme)
        phi_vec = solve_system(A, b)
        phi = phi_to_grid(phi_vec, Nx, Ny)
        residual = np.linalg.norm(A @ phi_vec - b)

    print(f"  Residual: {residual:.3e}")
    print(f"  Solution range: [{phi.min():.4f}, {phi.max():.4f}]")

    # Save plot with velocity vectors
    title = f'{case_name} ({Nx}×{Ny}, {scheme})'
    fname = os.path.join(save_dir, f'{case_name}_{Nx}x{Ny}_{scheme}.png')
    plot_phi(xc, yc, phi, title=title, fname=fname,
             ux_func=ux_func, uy_func=uy_func)

    return xc, yc, phi


# ==================== CONVERGENCE STUDY ====================
def convergence_study(ux_func, uy_func, scheme, save_dir, case_name):
    """Perform convergence study using three mesh levels"""
    print(f"\n{'=' * 70}")
    print(f"CONVERGENCE STUDY: {case_name} - {scheme.upper()} SCHEME")
    print(f"{'=' * 70}")

    results = {}
    for Nx, Ny in meshes:
        xc, yc, phi = run_case(Nx, Ny, ux_func, uy_func, scheme, save_dir,
                               f"{case_name}_{scheme}")
        results[(Nx, Ny)] = (xc, yc, phi)

    # Use finest grid as reference
    finest = meshes[-1]
    xref, yref, phiref = results[finest]

    # Compute errors between coarse and medium grids
    Nc = meshes[0]
    Nf = meshes[1]
    xc, yc, phic = results[Nc]
    xf, yf, phif = results[Nf]

    ec = compute_error_norm(phic, xc, yc, phiref, xref, yref)
    ef = compute_error_norm(phif, xf, yf, phiref, xref, yref)

    # Compute order of convergence
    hx_c = Lx / Nc[0]
    hx_f = Lx / Nf[0]
    order = np.log(ec / ef) / np.log(hx_c / hx_f)

    print(f"\nConvergence Results:")
    print(f"  Coarse mesh: {Nc[0]}×{Nc[1]}, h = {hx_c:.6f}, Error = {ec:.6e}")
    print(f"  Fine mesh:   {Nf[0]}×{Nf[1]}, h = {hx_f:.6f}, Error = {ef:.6e}")
    print(f"  Order of convergence: {order:.4f}")

    return order, ec, ef


# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    # ========== PROBLEM 2: Constant velocity, Γ = 0 ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 2: Constant velocity (u=2, v=2) with Γ=0")
    print("#" * 70)

    gamma = 0.0
    ux_func, uy_func = velocity_constant(2.0, 2.0)

    # Test central difference
    run_case(80, 80, ux_func, uy_func, 'central', 'results_problem2', 'const_vel_gamma0')

    # Test upwind
    run_case(80, 80, ux_func, uy_func, 'upwind', 'results_problem2', 'const_vel_gamma0')

    print("\n*** Check for oscillations or instabilities in the plots ***")

    # ========== PROBLEM 3: Constant velocity, Γ = 5 ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 3: Constant velocity (u=2, v=2) with Γ=5")
    print("#" * 70)

    gamma = 5.0

    # Test central difference
    run_case(80, 80, ux_func, uy_func, 'central', 'results_problem3', 'const_vel_gamma5')

    # Test upwind
    run_case(80, 80, ux_func, uy_func, 'upwind', 'results_problem3', 'const_vel_gamma5')

    print("\n*** Compare stability with Problem 2 ***")

    # ========== PROBLEM 4: Rotational velocity, Γ = 5, Convergence Study ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 4: Rotational velocity field with Γ=5")
    print("#" * 70)

    gamma = 5.0
    ux_func, uy_func = velocity_rotational()

    # Create velocity field visualization for rotational flow
    print("\nCreating rotational velocity field visualization...")
    xc_vis, yc_vis, _, _ = build_grid(12, 12)
    plot_velocity_field(xc_vis, yc_vis, ux_func, uy_func,
                        'Velocity Field: Rotational (u = -r sin θ, v = r cos θ)',
                        'results_problem4/velocity_field_rotational.png')

    # Convergence study for central difference
    order_central, ec_central, ef_central = convergence_study(
        ux_func, uy_func, 'central', 'results_problem4', 'rotational'
    )

    # Convergence study for upwind
    order_upwind, ec_upwind, ef_upwind = convergence_study(
        ux_func, uy_func, 'upwind', 'results_problem4', 'rotational'
    )

    # Summary
    print("\n" + "=" * 70)
    print("FINAL CONVERGENCE SUMMARY (Problems 2-4)")
    print("=" * 70)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind Scheme      - Order of Convergence: {order_upwind:.4f}")
    print("=" * 70)

    # ========== BONUS: QUICK scheme with rotational velocity ==========
    print("\n" + "#" * 70)
    print("# BONUS: QUICK SCHEME - Rotational velocity field with Γ=5")
    print("#" * 70)

    # Convergence study for QUICK
    order_quick, ec_quick, ef_quick = convergence_study(
        ux_func, uy_func, 'quick', 'results_bonus', 'rotational'
    )

    # Final comparison
    print("\n" + "=" * 70)
    print("COMPLETE CONVERGENCE SUMMARY (Including BONUS)")
    print("=" * 70)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind Scheme      - Order of Convergence: {order_upwind:.4f}")
    print(f"QUICK Scheme       - Order of Convergence: {order_quick:.4f}")
    print("=" * 70)
    print("\nExpected theoretical orders:")
    print("  - Central Difference: ~2.0 (second-order)")
    print("  - Upwind:            ~1.0 (first-order)")
    print("  - QUICK:             ~2.0-3.0 (between second and third order)")
    print("=" * 70)

    print("\nAll results saved to:")
    print("  - results_problem2/")
    print("  - results_problem3/")
    print("  - results_problem4/")
    print("  - results_bonus/ (QUICK scheme)")
    print("=" * 70)
, fontsize = 12,
rotation = 90, va = 'center', ha = 'right')
ax.text(1.05, 0.5, r'$\phi_b = 0


def plot_diagonal_comparison(results_dict, save_dir, case_name, gamma_val):
    """
    Create a comparison plot along the diagonal like Figure 5.15.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys like '80_central', '80_upwind', etc.
        Values are tuples of (xc, yc, phi)
    save_dir : str
        Directory to save the plot
    case_name : str
        Name for the plot file
    gamma_val : float
        Value of gamma for the title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define line styles and colors
    styles = {
        'central': {'linestyle': ':', 'linewidth': 2, 'color': 'black'},
        'upwind': {'linestyle': '-', 'linewidth': 2, 'color': 'black'}
    }

    resolutions = [80, 160, 320]

    for scheme in ['upwind', 'central']:
        for res in resolutions:
            key = f'{res}_{scheme}'
            if key not in results_dict:
                continue

            xc, yc, phi = results_dict[key]

            # Extract values along diagonal (x = y)
            # For each cell center, find the closest point to the diagonal
            nx, ny = len(xc), len(yc)
            X, Y = np.meshgrid(xc, yc)

            # Calculate distance along diagonal: d = sqrt(x^2 + y^2) for x=y line
            # For square domain, diagonal distance = sqrt(2) * x
            diagonal_distance = []
            phi_diagonal = []

            # Extract values close to the diagonal (within tolerance)
            tolerance = max(Lx / nx, Ly / ny) * 1.5  # Within 1.5 cell widths

            for j in range(ny):
                for i in range(nx):
                    x_val = X[j, i]
                    y_val = Y[j, i]

                    # Check if point is close to diagonal (x ≈ y)
                    if abs(x_val - y_val) < tolerance:
                        # Distance along diagonal from origin
                        dist = np.sqrt(x_val ** 2 + y_val ** 2)
                        diagonal_distance.append(dist)
                        phi_diagonal.append(phi[j, i])

            # Sort by distance
            sorted_indices = np.argsort(diagonal_distance)
            diagonal_distance = np.array(diagonal_distance)[sorted_indices]
            phi_diagonal = np.array(phi_diagonal)[sorted_indices]

            # Plot
            if scheme == 'upwind':
                if res == 80:
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='--', linewidth=2.5, color='black', label=label)
                elif res == 160:
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='-', linewidth=2, color='black', label=label)
                else:  # 320
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='-', linewidth=1.5, color='gray', label=label)
            else:  # central
                if res == 160:
                    label = f'Central {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle=':', linewidth=2.5, color='black', label=label)

    ax.set_xlabel('Distance along diagonal X = Y (m)', fontsize=12)
    ax.set_ylabel('φ', fontsize=12)
    ax.set_title(f'Solution along diagonal (Γ={gamma_val})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim(0, np.sqrt(2) * Lx)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'{case_name}_diagonal_comparison.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Diagonal comparison plot saved to: {fname}")


# ==================== CONVERGENCE ANALYSIS ====================
def compute_error_norm(phi_coarse, x_c, y_c, phi_ref, x_ref, y_ref):
    """
    Compute L2 error norm between coarse solution and reference solution.
    Uses bilinear interpolation to compare at coarse grid points.
    """
    # Create interpolator for reference solution
    interp_func = RegularGridInterpolator(
        (y_ref, x_ref),
        phi_ref,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Create meshgrid for coarse solution
    Yc, Xc = np.meshgrid(y_c, x_c, indexing='ij')
    points = np.column_stack([Yc.ravel(), Xc.ravel()])

    # Interpolate reference solution to coarse grid points
    phi_ref_on_coarse = interp_func(points).reshape(phi_coarse.shape)

    # Compute L2 error norm
    diff = phi_ref_on_coarse - phi_coarse
    error = np.sqrt(np.mean(diff ** 2))

    return error


# ==================== RUN SINGLE CASE ====================
def run_case(Nx, Ny, ux_func, uy_func, scheme, save_dir, case_name):
    """Run a single simulation case"""
    print(f"\nRunning {case_name}: {Nx}×{Ny} grid, {scheme} scheme...")
    os.makedirs(save_dir, exist_ok=True)

    # Assemble and solve
    A, b, xc, yc = assemble_system(Nx, Ny, ux_func, uy_func, scheme)
    phi_vec = solve_system(A, b)
    phi = phi_to_grid(phi_vec, Nx, Ny)

    # Check residual
    residual = np.linalg.norm(A @ phi_vec - b)
    print(f"  Residual: {residual:.3e}")
    print(f"  Solution range: [{phi.min():.4f}, {phi.max():.4f}]")

    # Save plot
    title = f'{case_name} ({Nx}×{Ny}, {scheme})'
    fname = os.path.join(save_dir, f'{case_name}_{Nx}x{Ny}_{scheme}.png')
    plot_phi(xc, yc, phi, title=title, fname=fname)

    return xc, yc, phi


# ==================== CONVERGENCE STUDY ====================
def convergence_study(ux_func, uy_func, scheme, save_dir, case_name):
    """Perform convergence study using three mesh levels"""
    print(f"\n{'=' * 70}")
    print(f"CONVERGENCE STUDY: {case_name} - {scheme.upper()} SCHEME")
    print(f"{'=' * 70}")

    results = {}
    for Nx, Ny in meshes:
        xc, yc, phi = run_case(Nx, Ny, ux_func, uy_func, scheme, save_dir,
                               f"{case_name}_{scheme}")
        results[(Nx, Ny)] = (xc, yc, phi)

    # Use finest grid as reference
    finest = meshes[-1]
    xref, yref, phiref = results[finest]

    # Compute errors between coarse and medium grids
    Nc = meshes[0]
    Nf = meshes[1]
    xc, yc, phic = results[Nc]
    xf, yf, phif = results[Nf]

    ec = compute_error_norm(phic, xc, yc, phiref, xref, yref)
    ef = compute_error_norm(phif, xf, yf, phiref, xref, yref)

    # Compute order of convergence
    hx_c = Lx / Nc[0]
    hx_f = Lx / Nf[0]
    order = np.log(ec / ef) / np.log(hx_c / hx_f)

    print(f"\nConvergence Results:")
    print(f"  Coarse mesh: {Nc[0]}×{Nc[1]}, h = {hx_c:.6f}, Error = {ec:.6e}")
    print(f"  Fine mesh:   {Nf[0]}×{Nf[1]}, h = {hx_f:.6f}, Error = {ef:.6e}")
    print(f"  Order of convergence: {order:.4f}")

    return order, ec, ef


# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    # ========== PROBLEM 2: Constant velocity, Γ = 0 ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 2: Constant velocity (u=2, v=2) with Γ=0")
    print("#" * 70)

    gamma = 0.0
    ux_func, uy_func = velocity_constant(2.0, 2.0)

    # Test central difference
    run_case(80, 80, ux_func, uy_func, 'central', 'results_problem2', 'const_vel_gamma0')

    # Test upwind
    run_case(80, 80, ux_func, uy_func, 'upwind', 'results_problem2', 'const_vel_gamma0')

    print("\n*** Check for oscillations or instabilities in the plots ***")

    # ========== PROBLEM 3: Constant velocity, Γ = 5 ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 3: Constant velocity (u=2, v=2) with Γ=5")
    print("#" * 70)

    gamma = 5.0

    # Test central difference
    run_case(80, 80, ux_func, uy_func, 'central', 'results_problem3', 'const_vel_gamma5')

    # Test upwind
    run_case(80, 80, ux_func, uy_func, 'upwind', 'results_problem3', 'const_vel_gamma5')

    print("\n*** Compare stability with Problem 2 ***")

    # ========== PROBLEM 4: Rotational velocity, Γ = 5, Convergence Study ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 4: Rotational velocity field with Γ=5")
    print("#" * 70)

    gamma = 5.0
    ux_func, uy_func = velocity_rotational()

    # Convergence study for central difference
    order_central, ec_central, ef_central = convergence_study(
        ux_func, uy_func, 'central', 'results_problem4', 'rotational'
    )

    # Convergence study for upwind
    order_upwind, ec_upwind, ef_upwind = convergence_study(
        ux_func, uy_func, 'upwind', 'results_problem4', 'rotational'
    )

    # Summary
    print("\n" + "=" * 70)
    print("FINAL CONVERGENCE SUMMARY (Problems 2-4)")
    print("=" * 70)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind Scheme      - Order of Convergence: {order_upwind:.4f}")
    print("=" * 70)

    # ========== BONUS: QUICK scheme with rotational velocity ==========
    print("\n" + "#" * 70)
    print("# BONUS: QUICK SCHEME - Rotational velocity field with Γ=5")
    print("#" * 70)

    # Convergence study for QUICK
    order_quick, ec_quick, ef_quick = convergence_study(
        ux_func, uy_func, 'quick', 'results_bonus', 'rotational'
    )

    # Final comparison
    print("\n" + "=" * 70)
    print("COMPLETE CONVERGENCE SUMMARY (Including BONUS)")
    print("=" * 70)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind Scheme      - Order of Convergence: {order_upwind:.4f}")
    print(f"QUICK Scheme       - Order of Convergence: {order_quick:.4f}")
    print("=" * 70)
    print("\nExpected theoretical orders:")
    print("  - Central Difference: ~2.0 (second-order)")
    print("  - Upwind:            ~1.0 (first-order)")
    print("  - QUICK:             ~2.0-3.0 (between second and third order)")
    print("=" * 70)

    print("\nAll results saved to:")
    print("  - results_problem2/")
    print("  - results_problem3/")
    print("  - results_problem4/")
    print("  - results_bonus/ (QUICK scheme)")
    print("=" * 70)
, fontsize = 12,
rotation = 90, va = 'center', ha = 'left')
ax.text(0.5, -0.05, r'$\phi_b = 0


def plot_diagonal_comparison(results_dict, save_dir, case_name, gamma_val):
    """
    Create a comparison plot along the diagonal like Figure 5.15.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys like '80_central', '80_upwind', etc.
        Values are tuples of (xc, yc, phi)
    save_dir : str
        Directory to save the plot
    case_name : str
        Name for the plot file
    gamma_val : float
        Value of gamma for the title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define line styles and colors
    styles = {
        'central': {'linestyle': ':', 'linewidth': 2, 'color': 'black'},
        'upwind': {'linestyle': '-', 'linewidth': 2, 'color': 'black'}
    }

    resolutions = [80, 160, 320]

    for scheme in ['upwind', 'central']:
        for res in resolutions:
            key = f'{res}_{scheme}'
            if key not in results_dict:
                continue

            xc, yc, phi = results_dict[key]

            # Extract values along diagonal (x = y)
            # For each cell center, find the closest point to the diagonal
            nx, ny = len(xc), len(yc)
            X, Y = np.meshgrid(xc, yc)

            # Calculate distance along diagonal: d = sqrt(x^2 + y^2) for x=y line
            # For square domain, diagonal distance = sqrt(2) * x
            diagonal_distance = []
            phi_diagonal = []

            # Extract values close to the diagonal (within tolerance)
            tolerance = max(Lx / nx, Ly / ny) * 1.5  # Within 1.5 cell widths

            for j in range(ny):
                for i in range(nx):
                    x_val = X[j, i]
                    y_val = Y[j, i]

                    # Check if point is close to diagonal (x ≈ y)
                    if abs(x_val - y_val) < tolerance:
                        # Distance along diagonal from origin
                        dist = np.sqrt(x_val ** 2 + y_val ** 2)
                        diagonal_distance.append(dist)
                        phi_diagonal.append(phi[j, i])

            # Sort by distance
            sorted_indices = np.argsort(diagonal_distance)
            diagonal_distance = np.array(diagonal_distance)[sorted_indices]
            phi_diagonal = np.array(phi_diagonal)[sorted_indices]

            # Plot
            if scheme == 'upwind':
                if res == 80:
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='--', linewidth=2.5, color='black', label=label)
                elif res == 160:
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='-', linewidth=2, color='black', label=label)
                else:  # 320
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='-', linewidth=1.5, color='gray', label=label)
            else:  # central
                if res == 160:
                    label = f'Central {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle=':', linewidth=2.5, color='black', label=label)

    ax.set_xlabel('Distance along diagonal X = Y (m)', fontsize=12)
    ax.set_ylabel('φ', fontsize=12)
    ax.set_title(f'Solution along diagonal (Γ={gamma_val})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim(0, np.sqrt(2) * Lx)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'{case_name}_diagonal_comparison.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Diagonal comparison plot saved to: {fname}")


# ==================== CONVERGENCE ANALYSIS ====================
def compute_error_norm(phi_coarse, x_c, y_c, phi_ref, x_ref, y_ref):
    """
    Compute L2 error norm between coarse solution and reference solution.
    Uses bilinear interpolation to compare at coarse grid points.
    """
    # Create interpolator for reference solution
    interp_func = RegularGridInterpolator(
        (y_ref, x_ref),
        phi_ref,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Create meshgrid for coarse solution
    Yc, Xc = np.meshgrid(y_c, x_c, indexing='ij')
    points = np.column_stack([Yc.ravel(), Xc.ravel()])

    # Interpolate reference solution to coarse grid points
    phi_ref_on_coarse = interp_func(points).reshape(phi_coarse.shape)

    # Compute L2 error norm
    diff = phi_ref_on_coarse - phi_coarse
    error = np.sqrt(np.mean(diff ** 2))

    return error


# ==================== RUN SINGLE CASE ====================
def run_case(Nx, Ny, ux_func, uy_func, scheme, save_dir, case_name):
    """Run a single simulation case"""
    print(f"\nRunning {case_name}: {Nx}×{Ny} grid, {scheme} scheme...")
    os.makedirs(save_dir, exist_ok=True)

    # Assemble and solve
    A, b, xc, yc = assemble_system(Nx, Ny, ux_func, uy_func, scheme)
    phi_vec = solve_system(A, b)
    phi = phi_to_grid(phi_vec, Nx, Ny)

    # Check residual
    residual = np.linalg.norm(A @ phi_vec - b)
    print(f"  Residual: {residual:.3e}")
    print(f"  Solution range: [{phi.min():.4f}, {phi.max():.4f}]")

    # Save plot
    title = f'{case_name} ({Nx}×{Ny}, {scheme})'
    fname = os.path.join(save_dir, f'{case_name}_{Nx}x{Ny}_{scheme}.png')
    plot_phi(xc, yc, phi, title=title, fname=fname)

    return xc, yc, phi


# ==================== CONVERGENCE STUDY ====================
def convergence_study(ux_func, uy_func, scheme, save_dir, case_name):
    """Perform convergence study using three mesh levels"""
    print(f"\n{'=' * 70}")
    print(f"CONVERGENCE STUDY: {case_name} - {scheme.upper()} SCHEME")
    print(f"{'=' * 70}")

    results = {}
    for Nx, Ny in meshes:
        xc, yc, phi = run_case(Nx, Ny, ux_func, uy_func, scheme, save_dir,
                               f"{case_name}_{scheme}")
        results[(Nx, Ny)] = (xc, yc, phi)

    # Use finest grid as reference
    finest = meshes[-1]
    xref, yref, phiref = results[finest]

    # Compute errors between coarse and medium grids
    Nc = meshes[0]
    Nf = meshes[1]
    xc, yc, phic = results[Nc]
    xf, yf, phif = results[Nf]

    ec = compute_error_norm(phic, xc, yc, phiref, xref, yref)
    ef = compute_error_norm(phif, xf, yf, phiref, xref, yref)

    # Compute order of convergence
    hx_c = Lx / Nc[0]
    hx_f = Lx / Nf[0]
    order = np.log(ec / ef) / np.log(hx_c / hx_f)

    print(f"\nConvergence Results:")
    print(f"  Coarse mesh: {Nc[0]}×{Nc[1]}, h = {hx_c:.6f}, Error = {ec:.6e}")
    print(f"  Fine mesh:   {Nf[0]}×{Nf[1]}, h = {hx_f:.6f}, Error = {ef:.6e}")
    print(f"  Order of convergence: {order:.4f}")

    return order, ec, ef


# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    # ========== PROBLEM 2: Constant velocity, Γ = 0 ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 2: Constant velocity (u=2, v=2) with Γ=0")
    print("#" * 70)

    gamma = 0.0
    ux_func, uy_func = velocity_constant(2.0, 2.0)

    # Test central difference
    run_case(80, 80, ux_func, uy_func, 'central', 'results_problem2', 'const_vel_gamma0')

    # Test upwind
    run_case(80, 80, ux_func, uy_func, 'upwind', 'results_problem2', 'const_vel_gamma0')

    print("\n*** Check for oscillations or instabilities in the plots ***")

    # ========== PROBLEM 3: Constant velocity, Γ = 5 ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 3: Constant velocity (u=2, v=2) with Γ=5")
    print("#" * 70)

    gamma = 5.0

    # Test central difference
    run_case(80, 80, ux_func, uy_func, 'central', 'results_problem3', 'const_vel_gamma5')

    # Test upwind
    run_case(80, 80, ux_func, uy_func, 'upwind', 'results_problem3', 'const_vel_gamma5')

    print("\n*** Compare stability with Problem 2 ***")

    # ========== PROBLEM 4: Rotational velocity, Γ = 5, Convergence Study ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 4: Rotational velocity field with Γ=5")
    print("#" * 70)

    gamma = 5.0
    ux_func, uy_func = velocity_rotational()

    # Convergence study for central difference
    order_central, ec_central, ef_central = convergence_study(
        ux_func, uy_func, 'central', 'results_problem4', 'rotational'
    )

    # Convergence study for upwind
    order_upwind, ec_upwind, ef_upwind = convergence_study(
        ux_func, uy_func, 'upwind', 'results_problem4', 'rotational'
    )

    # Summary
    print("\n" + "=" * 70)
    print("FINAL CONVERGENCE SUMMARY (Problems 2-4)")
    print("=" * 70)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind Scheme      - Order of Convergence: {order_upwind:.4f}")
    print("=" * 70)

    # ========== BONUS: QUICK scheme with rotational velocity ==========
    print("\n" + "#" * 70)
    print("# BONUS: QUICK SCHEME - Rotational velocity field with Γ=5")
    print("#" * 70)

    # Convergence study for QUICK
    order_quick, ec_quick, ef_quick = convergence_study(
        ux_func, uy_func, 'quick', 'results_bonus', 'rotational'
    )

    # Final comparison
    print("\n" + "=" * 70)
    print("COMPLETE CONVERGENCE SUMMARY (Including BONUS)")
    print("=" * 70)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind Scheme      - Order of Convergence: {order_upwind:.4f}")
    print(f"QUICK Scheme       - Order of Convergence: {order_quick:.4f}")
    print("=" * 70)
    print("\nExpected theoretical orders:")
    print("  - Central Difference: ~2.0 (second-order)")
    print("  - Upwind:            ~1.0 (first-order)")
    print("  - QUICK:             ~2.0-3.0 (between second and third order)")
    print("=" * 70)

    print("\nAll results saved to:")
    print("  - results_problem2/")
    print("  - results_problem3/")
    print("  - results_problem4/")
    print("  - results_bonus/ (QUICK scheme)")
    print("=" * 70)
, fontsize = 12,
ha = 'center', va = 'top')
ax.text(0.5, 1.05, r'$\phi_b = 100


def plot_diagonal_comparison(results_dict, save_dir, case_name, gamma_val):
    """
    Create a comparison plot along the diagonal like Figure 5.15.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys like '80_central', '80_upwind', etc.
        Values are tuples of (xc, yc, phi)
    save_dir : str
        Directory to save the plot
    case_name : str
        Name for the plot file
    gamma_val : float
        Value of gamma for the title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define line styles and colors
    styles = {
        'central': {'linestyle': ':', 'linewidth': 2, 'color': 'black'},
        'upwind': {'linestyle': '-', 'linewidth': 2, 'color': 'black'}
    }

    resolutions = [80, 160, 320]

    for scheme in ['upwind', 'central']:
        for res in resolutions:
            key = f'{res}_{scheme}'
            if key not in results_dict:
                continue

            xc, yc, phi = results_dict[key]

            # Extract values along diagonal (x = y)
            # For each cell center, find the closest point to the diagonal
            nx, ny = len(xc), len(yc)
            X, Y = np.meshgrid(xc, yc)

            # Calculate distance along diagonal: d = sqrt(x^2 + y^2) for x=y line
            # For square domain, diagonal distance = sqrt(2) * x
            diagonal_distance = []
            phi_diagonal = []

            # Extract values close to the diagonal (within tolerance)
            tolerance = max(Lx / nx, Ly / ny) * 1.5  # Within 1.5 cell widths

            for j in range(ny):
                for i in range(nx):
                    x_val = X[j, i]
                    y_val = Y[j, i]

                    # Check if point is close to diagonal (x ≈ y)
                    if abs(x_val - y_val) < tolerance:
                        # Distance along diagonal from origin
                        dist = np.sqrt(x_val ** 2 + y_val ** 2)
                        diagonal_distance.append(dist)
                        phi_diagonal.append(phi[j, i])

            # Sort by distance
            sorted_indices = np.argsort(diagonal_distance)
            diagonal_distance = np.array(diagonal_distance)[sorted_indices]
            phi_diagonal = np.array(phi_diagonal)[sorted_indices]

            # Plot
            if scheme == 'upwind':
                if res == 80:
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='--', linewidth=2.5, color='black', label=label)
                elif res == 160:
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='-', linewidth=2, color='black', label=label)
                else:  # 320
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='-', linewidth=1.5, color='gray', label=label)
            else:  # central
                if res == 160:
                    label = f'Central {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle=':', linewidth=2.5, color='black', label=label)

    ax.set_xlabel('Distance along diagonal X = Y (m)', fontsize=12)
    ax.set_ylabel('φ', fontsize=12)
    ax.set_title(f'Solution along diagonal (Γ={gamma_val})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim(0, np.sqrt(2) * Lx)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'{case_name}_diagonal_comparison.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Diagonal comparison plot saved to: {fname}")


# ==================== CONVERGENCE ANALYSIS ====================
def compute_error_norm(phi_coarse, x_c, y_c, phi_ref, x_ref, y_ref):
    """
    Compute L2 error norm between coarse solution and reference solution.
    Uses bilinear interpolation to compare at coarse grid points.
    """
    # Create interpolator for reference solution
    interp_func = RegularGridInterpolator(
        (y_ref, x_ref),
        phi_ref,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Create meshgrid for coarse solution
    Yc, Xc = np.meshgrid(y_c, x_c, indexing='ij')
    points = np.column_stack([Yc.ravel(), Xc.ravel()])

    # Interpolate reference solution to coarse grid points
    phi_ref_on_coarse = interp_func(points).reshape(phi_coarse.shape)

    # Compute L2 error norm
    diff = phi_ref_on_coarse - phi_coarse
    error = np.sqrt(np.mean(diff ** 2))

    return error


# ==================== RUN SINGLE CASE ====================
def run_case(Nx, Ny, ux_func, uy_func, scheme, save_dir, case_name):
    """Run a single simulation case"""
    print(f"\nRunning {case_name}: {Nx}×{Ny} grid, {scheme} scheme...")
    os.makedirs(save_dir, exist_ok=True)

    # Assemble and solve
    A, b, xc, yc = assemble_system(Nx, Ny, ux_func, uy_func, scheme)
    phi_vec = solve_system(A, b)
    phi = phi_to_grid(phi_vec, Nx, Ny)

    # Check residual
    residual = np.linalg.norm(A @ phi_vec - b)
    print(f"  Residual: {residual:.3e}")
    print(f"  Solution range: [{phi.min():.4f}, {phi.max():.4f}]")

    # Save plot
    title = f'{case_name} ({Nx}×{Ny}, {scheme})'
    fname = os.path.join(save_dir, f'{case_name}_{Nx}x{Ny}_{scheme}.png')
    plot_phi(xc, yc, phi, title=title, fname=fname)

    return xc, yc, phi


# ==================== CONVERGENCE STUDY ====================
def convergence_study(ux_func, uy_func, scheme, save_dir, case_name):
    """Perform convergence study using three mesh levels"""
    print(f"\n{'=' * 70}")
    print(f"CONVERGENCE STUDY: {case_name} - {scheme.upper()} SCHEME")
    print(f"{'=' * 70}")

    results = {}
    for Nx, Ny in meshes:
        xc, yc, phi = run_case(Nx, Ny, ux_func, uy_func, scheme, save_dir,
                               f"{case_name}_{scheme}")
        results[(Nx, Ny)] = (xc, yc, phi)

    # Use finest grid as reference
    finest = meshes[-1]
    xref, yref, phiref = results[finest]

    # Compute errors between coarse and medium grids
    Nc = meshes[0]
    Nf = meshes[1]
    xc, yc, phic = results[Nc]
    xf, yf, phif = results[Nf]

    ec = compute_error_norm(phic, xc, yc, phiref, xref, yref)
    ef = compute_error_norm(phif, xf, yf, phiref, xref, yref)

    # Compute order of convergence
    hx_c = Lx / Nc[0]
    hx_f = Lx / Nf[0]
    order = np.log(ec / ef) / np.log(hx_c / hx_f)

    print(f"\nConvergence Results:")
    print(f"  Coarse mesh: {Nc[0]}×{Nc[1]}, h = {hx_c:.6f}, Error = {ec:.6e}")
    print(f"  Fine mesh:   {Nf[0]}×{Nf[1]}, h = {hx_f:.6f}, Error = {ef:.6e}")
    print(f"  Order of convergence: {order:.4f}")

    return order, ec, ef


# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    # ========== PROBLEM 2: Constant velocity, Γ = 0 ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 2: Constant velocity (u=2, v=2) with Γ=0")
    print("#" * 70)

    gamma = 0.0
    ux_func, uy_func = velocity_constant(2.0, 2.0)

    # Test central difference
    run_case(80, 80, ux_func, uy_func, 'central', 'results_problem2', 'const_vel_gamma0')

    # Test upwind
    run_case(80, 80, ux_func, uy_func, 'upwind', 'results_problem2', 'const_vel_gamma0')

    print("\n*** Check for oscillations or instabilities in the plots ***")

    # ========== PROBLEM 3: Constant velocity, Γ = 5 ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 3: Constant velocity (u=2, v=2) with Γ=5")
    print("#" * 70)

    gamma = 5.0

    # Test central difference
    run_case(80, 80, ux_func, uy_func, 'central', 'results_problem3', 'const_vel_gamma5')

    # Test upwind
    run_case(80, 80, ux_func, uy_func, 'upwind', 'results_problem3', 'const_vel_gamma5')

    print("\n*** Compare stability with Problem 2 ***")

    # ========== PROBLEM 4: Rotational velocity, Γ = 5, Convergence Study ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 4: Rotational velocity field with Γ=5")
    print("#" * 70)

    gamma = 5.0
    ux_func, uy_func = velocity_rotational()

    # Convergence study for central difference
    order_central, ec_central, ef_central = convergence_study(
        ux_func, uy_func, 'central', 'results_problem4', 'rotational'
    )

    # Convergence study for upwind
    order_upwind, ec_upwind, ef_upwind = convergence_study(
        ux_func, uy_func, 'upwind', 'results_problem4', 'rotational'
    )

    # Summary
    print("\n" + "=" * 70)
    print("FINAL CONVERGENCE SUMMARY (Problems 2-4)")
    print("=" * 70)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind Scheme      - Order of Convergence: {order_upwind:.4f}")
    print("=" * 70)

    # ========== BONUS: QUICK scheme with rotational velocity ==========
    print("\n" + "#" * 70)
    print("# BONUS: QUICK SCHEME - Rotational velocity field with Γ=5")
    print("#" * 70)

    # Convergence study for QUICK
    order_quick, ec_quick, ef_quick = convergence_study(
        ux_func, uy_func, 'quick', 'results_bonus', 'rotational'
    )

    # Final comparison
    print("\n" + "=" * 70)
    print("COMPLETE CONVERGENCE SUMMARY (Including BONUS)")
    print("=" * 70)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind Scheme      - Order of Convergence: {order_upwind:.4f}")
    print(f"QUICK Scheme       - Order of Convergence: {order_quick:.4f}")
    print("=" * 70)
    print("\nExpected theoretical orders:")
    print("  - Central Difference: ~2.0 (second-order)")
    print("  - Upwind:            ~1.0 (first-order)")
    print("  - QUICK:             ~2.0-3.0 (between second and third order)")
    print("=" * 70)

    print("\nAll results saved to:")
    print("  - results_problem2/")
    print("  - results_problem3/")
    print("  - results_problem4/")
    print("  - results_bonus/ (QUICK scheme)")
    print("=" * 70)
, fontsize = 12,
ha = 'center', va = 'bottom')

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_title(title, fontsize=12, fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')

# Draw domain boundary
from matplotlib.patches import Rectangle

rect = Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black',
facecolor = 'none')
ax.add_patch(rect)

plt.tight_layout()
plt.savefig(fname, dpi=300, bbox_inches='tight')
plt.close()


def plot_diagonal_comparison(results_dict, save_dir, case_name, gamma_val):
    """
    Create a comparison plot along the diagonal like Figure 5.15.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys like '80_central', '80_upwind', etc.
        Values are tuples of (xc, yc, phi)
    save_dir : str
        Directory to save the plot
    case_name : str
        Name for the plot file
    gamma_val : float
        Value of gamma for the title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define line styles and colors
    styles = {
        'central': {'linestyle': ':', 'linewidth': 2, 'color': 'black'},
        'upwind': {'linestyle': '-', 'linewidth': 2, 'color': 'black'}
    }

    resolutions = [80, 160, 320]

    for scheme in ['upwind', 'central']:
        for res in resolutions:
            key = f'{res}_{scheme}'
            if key not in results_dict:
                continue

            xc, yc, phi = results_dict[key]

            # Extract values along diagonal (x = y)
            # For each cell center, find the closest point to the diagonal
            nx, ny = len(xc), len(yc)
            X, Y = np.meshgrid(xc, yc)

            # Calculate distance along diagonal: d = sqrt(x^2 + y^2) for x=y line
            # For square domain, diagonal distance = sqrt(2) * x
            diagonal_distance = []
            phi_diagonal = []

            # Extract values close to the diagonal (within tolerance)
            tolerance = max(Lx / nx, Ly / ny) * 1.5  # Within 1.5 cell widths

            for j in range(ny):
                for i in range(nx):
                    x_val = X[j, i]
                    y_val = Y[j, i]

                    # Check if point is close to diagonal (x ≈ y)
                    if abs(x_val - y_val) < tolerance:
                        # Distance along diagonal from origin
                        dist = np.sqrt(x_val ** 2 + y_val ** 2)
                        diagonal_distance.append(dist)
                        phi_diagonal.append(phi[j, i])

            # Sort by distance
            sorted_indices = np.argsort(diagonal_distance)
            diagonal_distance = np.array(diagonal_distance)[sorted_indices]
            phi_diagonal = np.array(phi_diagonal)[sorted_indices]

            # Plot
            if scheme == 'upwind':
                if res == 80:
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='--', linewidth=2.5, color='black', label=label)
                elif res == 160:
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='-', linewidth=2, color='black', label=label)
                else:  # 320
                    label = f'Upwind {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle='-', linewidth=1.5, color='gray', label=label)
            else:  # central
                if res == 160:
                    label = f'Central {res}×{res}'
                    ax.plot(diagonal_distance, phi_diagonal,
                            linestyle=':', linewidth=2.5, color='black', label=label)

    ax.set_xlabel('Distance along diagonal X = Y (m)', fontsize=12)
    ax.set_ylabel('φ', fontsize=12)
    ax.set_title(f'Solution along diagonal (Γ={gamma_val})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim(0, np.sqrt(2) * Lx)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'{case_name}_diagonal_comparison.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Diagonal comparison plot saved to: {fname}")


# ==================== CONVERGENCE ANALYSIS ====================
def compute_error_norm(phi_coarse, x_c, y_c, phi_ref, x_ref, y_ref):
    """
    Compute L2 error norm between coarse solution and reference solution.
    Uses bilinear interpolation to compare at coarse grid points.
    """
    # Create interpolator for reference solution
    interp_func = RegularGridInterpolator(
        (y_ref, x_ref),
        phi_ref,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Create meshgrid for coarse solution
    Yc, Xc = np.meshgrid(y_c, x_c, indexing='ij')
    points = np.column_stack([Yc.ravel(), Xc.ravel()])

    # Interpolate reference solution to coarse grid points
    phi_ref_on_coarse = interp_func(points).reshape(phi_coarse.shape)

    # Compute L2 error norm
    diff = phi_ref_on_coarse - phi_coarse
    error = np.sqrt(np.mean(diff ** 2))

    return error


# ==================== RUN SINGLE CASE ====================
def run_case(Nx, Ny, ux_func, uy_func, scheme, save_dir, case_name):
    """Run a single simulation case"""
    print(f"\nRunning {case_name}: {Nx}×{Ny} grid, {scheme} scheme...")
    os.makedirs(save_dir, exist_ok=True)

    # Assemble and solve
    A, b, xc, yc = assemble_system(Nx, Ny, ux_func, uy_func, scheme)
    phi_vec = solve_system(A, b)
    phi = phi_to_grid(phi_vec, Nx, Ny)

    # Check residual
    residual = np.linalg.norm(A @ phi_vec - b)
    print(f"  Residual: {residual:.3e}")
    print(f"  Solution range: [{phi.min():.4f}, {phi.max():.4f}]")

    # Save plot
    title = f'{case_name} ({Nx}×{Ny}, {scheme})'
    fname = os.path.join(save_dir, f'{case_name}_{Nx}x{Ny}_{scheme}.png')
    plot_phi(xc, yc, phi, title=title, fname=fname)

    return xc, yc, phi


# ==================== CONVERGENCE STUDY ====================
def convergence_study(ux_func, uy_func, scheme, save_dir, case_name):
    """Perform convergence study using three mesh levels"""
    print(f"\n{'=' * 70}")
    print(f"CONVERGENCE STUDY: {case_name} - {scheme.upper()} SCHEME")
    print(f"{'=' * 70}")

    results = {}
    for Nx, Ny in meshes:
        xc, yc, phi = run_case(Nx, Ny, ux_func, uy_func, scheme, save_dir,
                               f"{case_name}_{scheme}")
        results[(Nx, Ny)] = (xc, yc, phi)

    # Use finest grid as reference
    finest = meshes[-1]
    xref, yref, phiref = results[finest]

    # Compute errors between coarse and medium grids
    Nc = meshes[0]
    Nf = meshes[1]
    xc, yc, phic = results[Nc]
    xf, yf, phif = results[Nf]

    ec = compute_error_norm(phic, xc, yc, phiref, xref, yref)
    ef = compute_error_norm(phif, xf, yf, phiref, xref, yref)

    # Compute order of convergence
    hx_c = Lx / Nc[0]
    hx_f = Lx / Nf[0]
    order = np.log(ec / ef) / np.log(hx_c / hx_f)

    print(f"\nConvergence Results:")
    print(f"  Coarse mesh: {Nc[0]}×{Nc[1]}, h = {hx_c:.6f}, Error = {ec:.6e}")
    print(f"  Fine mesh:   {Nf[0]}×{Nf[1]}, h = {hx_f:.6f}, Error = {ef:.6e}")
    print(f"  Order of convergence: {order:.4f}")

    return order, ec, ef


# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    # ========== PROBLEM 2: Constant velocity, Γ = 0 ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 2: Constant velocity (u=2, v=2) with Γ=0")
    print("#" * 70)

    gamma = 0.0
    ux_func, uy_func = velocity_constant(2.0, 2.0)

    # Test central difference
    run_case(80, 80, ux_func, uy_func, 'central', 'results_problem2', 'const_vel_gamma0')

    # Test upwind
    run_case(80, 80, ux_func, uy_func, 'upwind', 'results_problem2', 'const_vel_gamma0')

    print("\n*** Check for oscillations or instabilities in the plots ***")

    # ========== PROBLEM 3: Constant velocity, Γ = 5 ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 3: Constant velocity (u=2, v=2) with Γ=5")
    print("#" * 70)

    gamma = 5.0

    # Test central difference
    run_case(80, 80, ux_func, uy_func, 'central', 'results_problem3', 'const_vel_gamma5')

    # Test upwind
    run_case(80, 80, ux_func, uy_func, 'upwind', 'results_problem3', 'const_vel_gamma5')

    print("\n*** Compare stability with Problem 2 ***")

    # ========== PROBLEM 4: Rotational velocity, Γ = 5, Convergence Study ==========
    print("\n" + "#" * 70)
    print("# PROBLEM 4: Rotational velocity field with Γ=5")
    print("#" * 70)

    gamma = 5.0
    ux_func, uy_func = velocity_rotational()

    # Convergence study for central difference
    order_central, ec_central, ef_central = convergence_study(
        ux_func, uy_func, 'central', 'results_problem4', 'rotational'
    )

    # Convergence study for upwind
    order_upwind, ec_upwind, ef_upwind = convergence_study(
        ux_func, uy_func, 'upwind', 'results_problem4', 'rotational'
    )

    # Summary
    print("\n" + "=" * 70)
    print("FINAL CONVERGENCE SUMMARY (Problems 2-4)")
    print("=" * 70)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind Scheme      - Order of Convergence: {order_upwind:.4f}")
    print("=" * 70)

    # ========== BONUS: QUICK scheme with rotational velocity ==========
    print("\n" + "#" * 70)
    print("# BONUS: QUICK SCHEME - Rotational velocity field with Γ=5")
    print("#" * 70)

    # Convergence study for QUICK
    order_quick, ec_quick, ef_quick = convergence_study(
        ux_func, uy_func, 'quick', 'results_bonus', 'rotational'
    )

    # Final comparison
    print("\n" + "=" * 70)
    print("COMPLETE CONVERGENCE SUMMARY (Including BONUS)")
    print("=" * 70)
    print(f"Central Difference - Order of Convergence: {order_central:.4f}")
    print(f"Upwind Scheme      - Order of Convergence: {order_upwind:.4f}")
    print(f"QUICK Scheme       - Order of Convergence: {order_quick:.4f}")
    print("=" * 70)
    print("\nExpected theoretical orders:")
    print("  - Central Difference: ~2.0 (second-order)")
    print("  - Upwind:            ~1.0 (first-order)")
    print("  - QUICK:             ~2.0-3.0 (between second and third order)")
    print("=" * 70)

    print("\nAll results saved to:")
    print("  - results_problem2/")
    print("  - results_problem3/")
    print("  - results_problem4/")
    print("  - results_bonus/ (QUICK scheme)")
    print("=" * 70)
