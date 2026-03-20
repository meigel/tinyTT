import numpy as np
import tinytt._backend as tn
from tinytt.basis import FourierBasis, LegendreBasis
from tinytt.functional import FunctionalTT


def _make1(rng):
    c0 = tn.tensor(rng.rand(1, 4, 1).astype(np.float64))
    return [c0]


def _make2(rng, r=2):
    c0 = tn.tensor(rng.rand(1, 4, r).astype(np.float64))
    c1 = tn.tensor(rng.rand(r, 4, 1).astype(np.float64))
    return [c0, c1]


def _make3(rng, r=2):
    c0 = tn.tensor(rng.rand(1, 3, r).astype(np.float64))
    c1 = tn.tensor(rng.rand(r, 3, r).astype(np.float64))
    c2 = tn.tensor(rng.rand(r, 3, 1).astype(np.float64))
    return [c0, c1, c2]


def _make_rank1(rng, degree, dims):
    return [tn.tensor(rng.rand(1, degree + 1, 1).astype(np.float64)) for _ in range(dims)]


def _make_vector_rank1(rng, degree=3, output_dim=2):
    c0 = tn.tensor(rng.rand(output_dim, degree + 1, 1).astype(np.float64))
    c1 = tn.tensor(rng.rand(1, degree + 1, 1).astype(np.float64))
    return [c0, c1]


def test_functional_tt_construct():
    rng = np.random.RandomState(0)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    ftt = FunctionalTT(_make2(rng), bases)
    assert ftt.d == 2
    assert ftt.num_features == [4, 4]


def test_functional_tt_call_single():
    rng = np.random.RandomState(0)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    ftt = FunctionalTT(_make2(rng), bases)
    x = tn.tensor([0.5, -0.3], dtype=tn.float64)
    out = ftt(x)
    assert out.shape == ()


def test_functional_tt_call_batch():
    rng = np.random.RandomState(0)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    ftt = FunctionalTT(_make2(rng), bases)
    x = tn.tensor([[0.5, -0.3], [0.1, 0.2], [-0.1, 0.4]], dtype=tn.float64)
    out = ftt(x)
    assert out.shape == (3,)


def test_functional_tt_call_batch_size_one_preserves_batch():
    rng = np.random.RandomState(0)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    ftt = FunctionalTT(_make2(rng), bases)
    x = tn.tensor([[0.5, -0.3]], dtype=tn.float64)
    out = ftt(x)
    assert out.shape == (1,)


def test_functional_tt_1d_vs_dense():
    rng = np.random.RandomState(123)
    bases = [LegendreBasis(degree=3)]
    core = _make1(rng)[0]
    ftt = FunctionalTT([core], bases)
    x = tn.tensor([[0.5], [-0.3], [0.0]], dtype=tn.float64)
    out_ftt = ftt(x).numpy()
    phi = bases[0](x[:, 0]).numpy()
    out_dense = phi @ core.numpy()[0, :, 0]
    np.testing.assert_allclose(out_ftt, out_dense, atol=1e-10)


def test_functional_tt_vs_dense():
    rng = np.random.RandomState(123)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    c0 = tn.tensor(rng.rand(1, 4, 1).astype(np.float64))
    c1 = tn.tensor(rng.rand(1, 4, 1).astype(np.float64))
    ftt = FunctionalTT([c0, c1], bases)
    x = tn.tensor([[0.5, -0.3], [0.0, 0.0]], dtype=tn.float64)
    out_ftt = ftt(x).numpy()
    phi0 = bases[0](x[:, 0]).numpy()
    phi1 = bases[1](x[:, 1]).numpy()
    g0 = c0.numpy()[0, :, 0]
    g1 = c1.numpy()[0, :, 0]
    out_dense = np.sum(phi0 * g0, axis=1) * np.sum(phi1 * g1, axis=1)
    np.testing.assert_allclose(out_ftt, out_dense, atol=1e-10)


def test_functional_tt_rank_one_five_dim_vs_dense():
    rng = np.random.RandomState(7)
    degree = 2
    dims = 5
    bases = [LegendreBasis(degree=degree) for _ in range(dims)]
    cores = _make_rank1(rng, degree, dims)
    ftt = FunctionalTT(cores, bases)
    x = tn.tensor(rng.uniform(-0.8, 0.8, size=(4, dims)), dtype=tn.float64)
    out_ftt = ftt(x).numpy()

    out_dense = np.ones(x.shape[0], dtype=np.float64)
    for i in range(dims):
        phi = bases[i](x[:, i]).numpy()
        out_dense *= phi @ cores[i].numpy()[0, :, 0]

    np.testing.assert_allclose(out_ftt, out_dense, atol=1e-10)


def test_functional_tt_grad_shape():
    rng = np.random.RandomState(0)
    bases = [LegendreBasis(degree=2) for _ in range(3)]
    ftt = FunctionalTT(_make3(rng), bases)
    x = tn.tensor([[0.5, -0.3, 0.1]], dtype=tn.float64)
    g = ftt.grad(x)
    assert g.shape == (1, 3)


def test_functional_tt_grad_numerical():
    rng = np.random.RandomState(42)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    ftt = FunctionalTT(_make2(rng), bases)
    x_np = np.array([[0.3, -0.2], [0.1, 0.4]], dtype=np.float64)
    x = tn.tensor(x_np, dtype=tn.float64)
    eps = 1e-6
    analytical = ftt.grad(x).numpy()
    num_grad = np.zeros((2, 2), dtype=np.float64)
    for b in range(2):
        for j in range(2):
            x_plus = x_np.copy()
            x_plus[b, j] += eps
            x_minus = x_np.copy()
            x_minus[b, j] -= eps
            vp = float(ftt(tn.tensor(x_plus, dtype=tn.float64)).numpy()[b])
            vm = float(ftt(tn.tensor(x_minus, dtype=tn.float64)).numpy()[b])
            num_grad[b, j] = (vp - vm) / (2 * eps)
    np.testing.assert_allclose(analytical, num_grad, atol=1e-4)


def test_functional_tt_jacobian_matches_grad_for_scalar_output():
    rng = np.random.RandomState(11)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    ftt = FunctionalTT(_make2(rng), bases)
    x = tn.tensor([[0.25, -0.5], [0.1, 0.2]], dtype=tn.float64)
    np.testing.assert_allclose(ftt.jacobian(x).numpy(), ftt.grad(x).numpy(), atol=1e-10)


def test_functional_tt_vector_jacobian_dense():
    rng = np.random.RandomState(5)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    c0, c1 = _make_vector_rank1(rng, degree=3, output_dim=2)
    ftt = FunctionalTT([c0, c1], bases)
    x = tn.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=tn.float64)

    jac = ftt.jacobian(x).numpy()
    phi0 = bases[0](x[:, 0]).numpy()
    phi1 = bases[1](x[:, 1]).numpy()
    dphi0 = bases[0].grad(x[:, 0]).numpy()
    dphi1 = bases[1].grad(x[:, 1]).numpy()
    b = c1.numpy()[0, :, 0]

    expected = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
    for out_idx in range(2):
        a = c0.numpy()[out_idx, :, 0]
        expected[:, out_idx, 0] = (dphi0 @ a) * (phi1 @ b)
        expected[:, out_idx, 1] = (phi0 @ a) * (dphi1 @ b)

    np.testing.assert_allclose(jac, expected, atol=1e-10)


def test_functional_tt_vector_divergence_dense():
    rng = np.random.RandomState(6)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    c0, c1 = _make_vector_rank1(rng, degree=3, output_dim=2)
    ftt = FunctionalTT([c0, c1], bases)
    x = tn.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=tn.float64)

    jac = ftt.jacobian(x).numpy()
    div = ftt.divergence(x).numpy()
    expected = jac[:, 0, 0] + jac[:, 1, 1]
    np.testing.assert_allclose(div, expected, atol=1e-10)


def test_functional_tt_vector_laplace_shape():
    rng = np.random.RandomState(7)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    c0, c1 = _make_vector_rank1(rng, degree=3, output_dim=2)
    ftt = FunctionalTT([c0, c1], bases)
    x = tn.tensor([[0.5, -0.3], [0.1, 0.2]], dtype=tn.float64)
    lap = ftt.laplace(x)
    assert lap.shape == (2, 2)


def test_functional_tt_laplace_shape():
    rng = np.random.RandomState(0)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    ftt = FunctionalTT(_make2(rng), bases)
    x = tn.tensor([[0.5, -0.3], [0.1, 0.2]], dtype=tn.float64)
    lap = ftt.laplace(x)
    assert lap.shape == (2,)


def test_functional_tt_laplace_numerical():
    rng = np.random.RandomState(99)
    bases = [LegendreBasis(degree=3), LegendreBasis(degree=3)]
    ftt = FunctionalTT(_make2(rng), bases)
    x_np = np.array([[0.3, -0.2], [0.1, 0.4]], dtype=np.float64)
    x = tn.tensor(x_np, dtype=tn.float64)
    eps = 1e-5
    analytical = ftt.laplace(x).numpy()
    num_lap = np.zeros((2,), dtype=np.float64)
    for b in range(2):
        fc = float(ftt(tn.tensor(x_np, dtype=tn.float64)).numpy()[b])
        for j in range(2):
            x_plus = x_np.copy()
            x_minus = x_np.copy()
            x_plus[b, j] += eps
            x_minus[b, j] -= eps
            fp = float(ftt(tn.tensor(x_plus, dtype=tn.float64)).numpy()[b])
            fm = float(ftt(tn.tensor(x_minus, dtype=tn.float64)).numpy()[b])
            num_lap[b] += (fp - 2 * fc + fm) / (eps ** 2)
    np.testing.assert_allclose(analytical, num_lap, atol=1e-3)


def test_functional_tt_fourier():
    rng = np.random.RandomState(0)
    bases = [FourierBasis(num_terms=2), FourierBasis(num_terms=2)]
    c0 = tn.tensor(rng.rand(1, 5, 2).astype(np.float64))
    c1 = tn.tensor(rng.rand(2, 5, 1).astype(np.float64))
    ftt = FunctionalTT([c0, c1], bases)
    x = tn.tensor([[0.0, 0.0], [np.pi / 2, np.pi]], dtype=tn.float64)
    out = ftt(x)
    assert out.shape == (2,)
