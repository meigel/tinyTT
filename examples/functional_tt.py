import numpy as np

import tinytt as tt
import tinytt._backend as tn
from tinytt.basis import OrthogonalPolynomialBasis
from tinytt.functional import FunctionalTT
from tinytt.regression import als_continuity_fit, als_regression

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

# End-to-end vector field fit: fit values, then use divergence on the fitted model.
xy0, xy1 = np.meshgrid(np.linspace(-0.8, 0.8, 9), np.linspace(-0.7, 0.7, 8), indexing="ij")
train_xy = tn.tensor(np.column_stack([xy0.ravel(), xy1.ravel()]), dtype=tn.float64)

# Choose a rank-1 vector field so the example matches the model class exactly.
clean_field = tn.stack(
    [
        (1.0 + 0.5 * train_xy[:, 0]) * (1.0 - 0.25 * train_xy[:, 1]),
        (-0.2 + 0.5 * train_xy[:, 0]) * (1.0 - 0.25 * train_xy[:, 1]),
    ],
    dim=1,
)
noise = tn.tensor(
    0.01 * np.column_stack([
        np.sin(7.0 * train_xy.numpy()[:, 0] - 3.0 * train_xy.numpy()[:, 1]),
        np.cos(5.0 * train_xy.numpy()[:, 0] + 2.0 * train_xy.numpy()[:, 1]),
    ]),
    dtype=tn.float64,
)
observed_field = clean_field + noise
vector_fit = als_regression(train_xy, observed_field, bases, ranks=[1], sweeps=6)
value_rel_err = (tn.linalg.norm(vector_fit(train_xy) - clean_field) / tn.linalg.norm(clean_field)).numpy().item()
print("vector fit relative value error:", value_rel_err)

probe = tn.tensor([[0.1, -0.2], [-0.4, 0.3], [0.6, 0.2]], dtype=tn.float64)
probe_div = vector_fit.divergence(probe).numpy()
expected_div = (0.5 * (1.0 - 0.25 * probe[:, 1]) - 0.25 * (-0.2 + 0.5 * probe[:, 0])).numpy()
print("vector fit divergence:\n", probe_div)
print("expected divergence:\n", expected_div)
print("divergence residual:", np.linalg.norm(probe_div - expected_div))

# The fitted model remains a regular tinyTT FunctionalTT object.
print("vector fit ranks:", vector_fit.R)
print("vector fit output_dim:", vector_fit.output_dim)


# Stationary continuity fit: recover a one-dimensional vector field V from
# F_grad(x) * V(x) + div(V)(x) = y(x).
continuity_x = tn.tensor(np.linspace(-0.85, 0.85, 51), dtype=tn.float64).reshape(51, 1)
continuity_bases = [OrthogonalPolynomialBasis(degree=3, family="legendre")]
continuity_target = FunctionalTT(
    [tn.tensor([[[0.6], [-0.2], [0.15], [0.05]]], dtype=tn.float64)],
    continuity_bases,
)
continuity_f_grad = tn.tensor((0.4 + 0.3 * continuity_x.numpy()[:, 0])[:, None], dtype=tn.float64)
continuity_y = continuity_f_grad[:, 0] * continuity_target(continuity_x) + continuity_target.divergence(continuity_x)
continuity_fit = als_continuity_fit(continuity_x, continuity_y, continuity_f_grad, continuity_bases, sweeps=6)
continuity_value_err = (
    tn.linalg.norm(continuity_fit(continuity_x) - continuity_target(continuity_x))
    / tn.linalg.norm(continuity_target(continuity_x))
).numpy().item()
continuity_residual_err = (
    tn.linalg.norm(continuity_f_grad[:, 0] * continuity_fit(continuity_x) + continuity_fit.divergence(continuity_x) - continuity_y)
    / tn.linalg.norm(continuity_y)
).numpy().item()
print("continuity fit relative value error:", continuity_value_err)
print("continuity fit relative residual error:", continuity_residual_err)
