import numpy as np
import tinytt._backend as tn
from tinytt.basis import OrthogonalPolynomialBasis
from tinytt.functional import FunctionalTT
from tinytt.regression import als_regression

# One basis per input dimension.
bases = [
    OrthogonalPolynomialBasis(degree=3, family="legendre"),
    OrthogonalPolynomialBasis(degree=3, family="legendre"),
]

# Scalar-valued FunctionalTT: output_dim = 1 because the first TT rank is 1.
scalar_cores = [
    tn.tensor([[[1.0], [0.3], [-0.1], [0.2]]], dtype=tn.float64),
    tn.tensor([[[0.7], [-0.4], [0.2], [0.1]]], dtype=tn.float64),
]
scalar_model = FunctionalTT(scalar_cores, bases)

points = tn.tensor([[0.0, 0.0], [0.4, -0.3], [-0.2, 0.5]], dtype=tn.float64)
print("scalar values:\n", scalar_model(points).numpy())
print("scalar gradients:\n", scalar_model.grad(points).numpy())
print("scalar laplace:\n", scalar_model.laplace(points).numpy())

# Vector-valued FunctionalTT: output_dim = 2 because the first TT rank is 2.
vector_cores = [
    tn.tensor(
        [
            [[1.0], [0.0], [0.2], [0.0]],
            [[0.0], [1.0], [0.0], [-0.1]],
        ],
        dtype=tn.float64,
    ),
    tn.tensor([[[0.5], [0.2], [-0.3], [0.1]]], dtype=tn.float64),
]
vector_field = FunctionalTT(vector_cores, bases)
print("vector values:\n", vector_field(points).numpy())
print("jacobian shape:", vector_field.jacobian(points).shape)
print("divergence:\n", vector_field.divergence(points).numpy())

# ALS regression uses the same basis list and infers output_dim from Y.
train_x = tn.tensor(np.linspace(-0.8, 0.8, 21), dtype=tn.float64).reshape(21, 1)
train_bases = [OrthogonalPolynomialBasis(degree=3, family="legendre")]
train_y = 1.0 + 0.5 * train_x[:, 0] - 0.25 * train_x[:, 0] ** 2
fitted = als_regression(train_x, train_y, train_bases, sweeps=3)
fit_error = tn.linalg.norm(fitted(train_x) - train_y).numpy().item()
print("scalar fit residual:", fit_error)
