"""
darcy.py — 2D parametric Darcy flow solver.

  -∇·(a(x, y) ∇u) = f(x)   in Ω = [0, 1]²
                 u = 0      on ∂Ω

Random permeability: a(x, y) = a₀(x) + σ·Σₖ √λₖ φₖ(x) yₖ
(lognormal or affine model via KL expansion).

Requires: scikit-fem, scipy, numpy.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

from skfem import BilinearForm, ElementQuad1, InteriorBasis, LinearForm
from skfem import MeshQuad, condense
from skfem.helpers import dot, grad


# -------------------------------------------------------------------
#  FEM forms (compiled once at module load)
# -------------------------------------------------------------------

@LinearForm
def _unit_load(v, w):
    """Unit source term f(x) = 1."""
    return v


@BilinearForm
def _mass_form(u, v, w):
    return u * v


@BilinearForm
def _stiffness_form(u, v, w):
    return dot(grad(u), grad(v))


# -------------------------------------------------------------------
#  Darcy sampler
# -------------------------------------------------------------------

class DarcySampler:
    """Sparse FEM sampler for parametric 2D Darcy problems.

    Solves  -∇·(a(x,y)∇u) = 1  with homogeneous Dirichlet BC on [0,1]².

    The random permeability field uses a Karhunen–Loève expansion
    with eigenfunctions φₖ(x) = sin(π·k·x₁)·sin(π·k·x₂) and
    eigenvalues λₖ = 1/k² (k = 1, …, M).

    Two coefficient models:

        lognormal: a(x, y) = exp(σ · Σₖ √λₖ φₖ(x) yₖ)
        affine:    a(x, y) = 1 + σ · Σₖ √λₖ φₖ(x) yₖ

    Parameters
    ----------
    n_nodes : int
        Number of nodes per spatial dimension (mesh is n_nodes × n_nodes).
    mkl : int
        Number of KL expansion terms (parameter dimension).
    sigma : float
        Standard deviation of the random field.
    intorder : int
        Quadrature order for FEM assembly.
    coefficient_model : str
        'lognormal' or 'affine'.
    """

    def __init__(self, n_nodes=101, mkl=10, sigma=0.1, intorder=5,
                 coefficient_model='lognormal'):
        self.n_nodes = int(n_nodes)
        self.mkl = int(mkl)
        self.sigma = float(sigma)

        if coefficient_model not in ('lognormal', 'affine'):
            raise ValueError("coefficient_model must be 'lognormal' or 'affine'")
        self.coefficient_model = coefficient_model

        # KL expansion
        self.modes = np.arange(1, self.mkl + 1)
        self.lambdas = 1.0 / self.modes ** 2

        # Mesh and FE space
        xs = np.linspace(0, 1, self.n_nodes)
        mesh = MeshQuad.init_tensor(xs, xs)
        self.mesh = mesh.with_boundaries({
            'left': lambda x: x[0] == 0.0,
            'right': lambda x: x[0] == 1.0,
            'bottom': lambda x: x[1] == 0.0,
            'top': lambda x: x[1] == 1.0,
        })
        self.basis = InteriorBasis(self.mesh, ElementQuad1(),
                                    intorder=intorder)
        self.dofs = self.basis.get_dofs(list(self.mesh.boundaries.keys()))
        self.b_full = _unit_load.assemble(self.basis)

        # Mass / stiffness / Gram matrices
        self.mass = _mass_form.assemble(self.basis).tocsr()
        self.stiffness = _stiffness_form.assemble(self.basis).tocsr()
        self._gram = (self.mass + self.stiffness).tocsr()

    @property
    def gram(self) -> sp.csr_matrix:
        """Mass + stiffness matrix for H¹ energy-norm computation."""
        return self._gram

    @property
    def dof(self) -> int:
        """Number of degrees of freedom (interior nodes)."""
        return int(self.basis.N)

    # ------------------------------------------------------------------
    #  Solve
    # ------------------------------------------------------------------

    def solve(self, y: np.ndarray) -> np.ndarray:
        """Solve the Darcy PDE for one parametric sample.

        Parameters
        ----------
        y : ndarray of shape (mkl,)
            KL coefficient vector (entries typically in [-1, 1]).

        Returns
        -------
        u : ndarray of shape (dof,)
            Solution at FEM nodes (interior + boundary, with zeros
            on the boundary).
        """
        y = np.asarray(y, dtype=float).ravel()
        modes = self.modes
        lambdas = self.lambdas
        sigma = self.sigma

        # Build the parametric diffusion coefficient at quadrature points
        @BilinearForm
        def diffusion(u, v, w):
            x0, x1 = w.x
            field = np.zeros_like(x0)
            for idx, mode in enumerate(modes):
                field += (np.sqrt(lambdas[idx])
                          * np.sin(np.pi * mode * x0)
                          * np.sin(np.pi * mode * x1)
                          * y[idx])
            if self.coefficient_model == 'lognormal':
                coeff = np.exp(sigma * field)
            else:
                coeff = 1.0 + sigma * field
            return coeff * dot(grad(u), grad(v))

        A = diffusion.assemble(self.basis)
        A_c, b_c, x_full, interior = condense(A, self.b_full, D=self.dofs)
        x_full = np.asarray(x_full, dtype=float)
        x_full[interior] = spsolve(A_c.tocsr(), b_c)
        return x_full

    # ------------------------------------------------------------------
    #  Error metrics
    # ------------------------------------------------------------------

    def relative_l2_h1(self, err: np.ndarray,
                       ref: np.ndarray) -> tuple[float, float]:
        """Relative L² and H¹ (energy-norm) errors.

        Parameters
        ----------
        err : ndarray — error vector u_pred - u_true.
        ref : ndarray — reference solution u_true.

        Returns
        -------
        l2_err : float — relative L² error.
        h1_err : float — relative H¹ (energy-norm) error.
        """
        err = np.asarray(err, dtype=float)
        ref = np.asarray(ref, dtype=float)
        l2 = np.sqrt(float(err @ (self.mass @ err))
                     / float(ref @ (self.mass @ ref)))
        h1 = np.sqrt(float(err @ (self._gram @ err))
                     / float(ref @ (self._gram @ ref)))
        return float(l2), float(h1)

    # ------------------------------------------------------------------
    #  Batch data generation
    # ------------------------------------------------------------------

    def generate_dataset(self, n_samples: int, seed=42,
                         param_range=1.0) -> tuple[np.ndarray, np.ndarray]:
        """Generate a dataset of (Y, U) pairs.

        Parameters
        ----------
        n_samples : int — number of samples.
        seed : int — random seed.
        param_range : float — parameters drawn uniformly in
            [-param_range, param_range].

        Returns
        -------
        Y : ndarray (n_samples, mkl) — parameter vectors.
        U : ndarray (n_samples, dof) — PDE solutions.
        """
        rng = np.random.RandomState(seed)
        Y = rng.uniform(-param_range, param_range, size=(n_samples, self.mkl))
        U = np.zeros((n_samples, self.dof))
        for i in range(n_samples):
            U[i] = self.solve(Y[i])
        return Y, U
