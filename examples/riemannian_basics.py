"""
Riemannian operations on the fixed-rank TT manifold.

Demonstrates:
  - left/right orthogonalisation (gauge transforms that preserve the tensor),
  - mixed-canonical form centred at a chosen site,
  - projection of a dense tensor onto the tangent space at a TT,
  - one step of Riemannian gradient descent for low-rank approximation
    (gradient is projected, then retracted back to the manifold by rounding).
"""

import numpy as np

import tinytt as tt
from tinytt import riemannian as rm


def main():
    rng = np.random.RandomState(0)
    shape = [4, 4, 4, 4]
    ranks = [1, 2, 3, 2, 1]

    # A random rank-(2,3,2) TT and a random target TT (also rank-bounded).
    x = tt.random(shape, ranks)
    target = tt.random(shape, [1, 3, 4, 3, 1])

    # Orthogonalisations preserve the represented tensor (gauge transform).
    xL = rm.left_orthogonalize(x)
    xR = rm.right_orthogonalize(x)
    print("left-orth gauge error: ",
          float(np.linalg.norm(xL.full().numpy() - x.full().numpy())))
    print("right-orth gauge error:",
          float(np.linalg.norm(xR.full().numpy() - x.full().numpy())))

    # Riemannian gradient descent on f(x) = 0.5 * || x - target ||^2.
    # Euclidean gradient (x - target) is projected onto the tangent space at x
    # and used as a descent direction; retraction maps back onto rank-r TTs.
    # All operations are TT-native: nothing is expanded to dense.
    for it in range(15):
        grad_tt = rm.tt_add(x, rm.tt_scale(target, -1.0))
        eta = rm.riemannian_grad(x, grad_tt)
        x = rm.retract(x, eta, step=-0.5, rmax=max(ranks))
        diff = rm.tt_add(x, rm.tt_scale(target, -1.0))
        loss = 0.5 * float(diff.norm().numpy().item()) ** 2
        print(f"iter {it:2d}: rmax={max(x.R)}  loss={loss:.6f}")


if __name__ == "__main__":
    main()
