"""Dense-reference tests for the matrix-free TT manifold layer."""

from __future__ import annotations

import numpy as np

import tinytt as tt
import tinytt._backend as tn
from tinytt.manifold import (
    TTManifoldFrame,
    TTTangentBatch,
    projection_transport,
    transport_batch,
)


def _random_tt(d=3, n=3, r=2, seed=0):
    rng = np.random.default_rng(seed)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = [
        tn.tensor(
            rng.standard_normal((ranks[k], n, ranks[k + 1])),
            dtype=tn.float64,
        )
        for k in range(d)
    ]
    return tt.TT(cores)


def _dense(cores):
    arrays = [
        tn.to_numpy(core) if tn.is_tensor(core) else np.asarray(core)
        for core in cores
    ]
    value = arrays[0][0]
    for core in arrays[1:]:
        value = np.einsum("...a,anb->...nb", value, core)
    return value[..., 0]


def _dense_tangent_sum(frame, blocks):
    total = np.zeros(frame.modes)
    for site in range(frame.order):
        cores = []
        for k in range(frame.order):
            if k < site:
                cores.append(frame.left_cores[k])
            elif k == site:
                cores.append(blocks[k])
            else:
                cores.append(frame.right_cores[k])
        total += _dense(cores)
    return total


def _dense_projector_basis(frame):
    columns = []
    for site in range(frame.order):
        r_left = frame.ranks[site]
        mode = frame.modes[site]
        r_right = frame.ranks[site + 1]

        if site < frame.order - 1:
            core = tn.to_numpy(frame.left_cores[site]).reshape(
                r_left * mode, r_right
            )
            complete, _ = np.linalg.qr(core, mode="complete")
            complement = complete[:, r_right:]
            local_blocks = []
            for column in range(r_right):
                for basis_index in range(complement.shape[1]):
                    block = np.zeros((r_left * mode, r_right))
                    block[:, column] = complement[:, basis_index]
                    local_blocks.append(block.reshape(r_left, mode, r_right))
        else:
            local_blocks = [
                np.eye(r_left * mode * r_right)[index].reshape(
                    r_left, mode, r_right
                )
                for index in range(r_left * mode * r_right)
            ]

        for local in local_blocks:
            blocks = [
                np.zeros(
                    (
                        frame.ranks[k],
                        frame.modes[k],
                        frame.ranks[k + 1],
                    )
                )
                for k in range(frame.order)
            ]
            blocks[site] = local
            columns.append(_dense_tangent_sum(frame, blocks).reshape(-1))
    return np.stack(columns, axis=1)


def _apply_gauges(tensor, seed=0, *, orthogonal=True):
    rng = np.random.default_rng(seed)
    cores = [tn.to_numpy(core).copy() for core in tensor.cores]
    for k in range(len(cores) - 1):
        rank = cores[k].shape[2]
        left, _ = np.linalg.qr(rng.standard_normal((rank, rank)))
        right, _ = np.linalg.qr(rng.standard_normal((rank, rank)))
        singular_values = (
            np.ones(rank)
            if orthogonal
            else np.geomspace(0.5, 2.0, num=rank)
        )
        gauge = left @ np.diag(singular_values) @ right.T
        inverse = np.linalg.inv(gauge)
        cores[k] = np.einsum("lna,ab->lnb", cores[k], gauge)
        cores[k + 1] = np.einsum("ba,anr->bnr", inverse, cores[k + 1])
    return tt.TT([tn.tensor(core, dtype=tn.float64) for core in cores])


def test_frame_preserves_tensor_and_reports_dimension():
    tensor = _random_tt(d=4, n=3, r=2, seed=1)
    frame = TTManifoldFrame.from_tt(tensor)

    np.testing.assert_allclose(
        _dense(frame.left_cores),
        tn.to_numpy(tensor.full()),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        _dense(frame.right_cores),
        tn.to_numpy(tensor.full()),
        rtol=1e-10,
        atol=1e-10,
    )
    expected = sum(
        frame.ranks[k] * frame.modes[k] * frame.ranks[k + 1]
        for k in range(frame.order)
    ) - sum(rank**2 for rank in frame.ranks[1:-1])
    assert frame.tangent_dimension == expected
    assert frame.regularity().regular


def test_tangent_synthesis_isometry_and_rank_bound():
    tensor = _random_tt(d=5, n=3, r=2, seed=2)
    frame = TTManifoldFrame.from_tt(tensor)
    tangent = frame.random_tangent(seed=3)
    tangent_tt = tangent.to_tt()

    dense_reference = _dense_tangent_sum(frame, tangent.blocks)
    dense_synthesized = tn.to_numpy(tangent_tt.full())
    np.testing.assert_allclose(
        dense_synthesized, dense_reference, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        float(tn.to_numpy(tangent.norm()).item()),
        np.linalg.norm(dense_reference),
        rtol=1e-10,
        atol=1e-10,
    )
    assert tangent.gauge_residual() < 1e-10
    for rank, base_rank in zip(tangent_tt.R[1:-1], frame.ranks[1:-1]):
        assert rank <= 2 * base_rank


def test_affine_synthesis_is_exact():
    tensor = _random_tt(d=4, n=3, r=2, seed=4)
    frame = TTManifoldFrame.from_tt(tensor)
    tangent = frame.random_tangent(seed=5)
    step = 0.17

    dense_expected = tn.to_numpy(tensor.full()) + step * _dense_tangent_sum(
        frame, tangent.blocks
    )
    np.testing.assert_allclose(
        tn.to_numpy(tangent.affine_to_tt(step).full()),
        dense_expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_one_pass_projection_matches_independent_dense_oracle():
    base = _random_tt(d=3, n=3, r=2, seed=6)
    ambient = _random_tt(d=3, n=3, r=3, seed=7)
    frame = TTManifoldFrame.from_tt(base)
    projected = frame.project(ambient)

    basis = _dense_projector_basis(frame)
    np.testing.assert_allclose(
        basis.T @ basis,
        np.eye(frame.tangent_dimension),
        rtol=1e-10,
        atol=1e-10,
    )
    ambient_dense = tn.to_numpy(ambient.full()).reshape(-1)
    dense_oracle = (basis @ (basis.T @ ambient_dense)).reshape(frame.modes)
    np.testing.assert_allclose(
        tn.to_numpy(projected.to_tt().full()),
        dense_oracle,
        rtol=1e-9,
        atol=1e-9,
    )


def test_projector_idempotence_self_adjointness_and_gauge_invariance():
    base = _random_tt(d=3, n=3, r=2, seed=8)
    first = _random_tt(d=3, n=3, r=2, seed=9)
    second = _random_tt(d=3, n=3, r=2, seed=10)
    frame = TTManifoldFrame.from_tt(base)

    projected_first = frame.project(first)
    projected_second = frame.project(second)
    projected_twice = frame.project(projected_first.to_tt())
    np.testing.assert_allclose(
        tn.to_numpy(projected_twice.to_tt().full()),
        tn.to_numpy(projected_first.to_tt().full()),
        rtol=1e-9,
        atol=1e-9,
    )

    lhs = np.vdot(
        tn.to_numpy(projected_first.to_tt().full()),
        tn.to_numpy(second.full()),
    )
    rhs = np.vdot(
        tn.to_numpy(first.full()),
        tn.to_numpy(projected_second.to_tt().full()),
    )
    np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-9)

    gauged_frame = TTManifoldFrame.from_tt(
        _apply_gauges(base, seed=11)
    )
    np.testing.assert_allclose(
        tn.to_numpy(gauged_frame.project(first).to_tt().full()),
        tn.to_numpy(projected_first.to_tt().full()),
        rtol=1e-9,
        atol=1e-9,
    )
    nonorthogonal_frame = TTManifoldFrame.from_tt(
        _apply_gauges(base, seed=12, orthogonal=False)
    )
    np.testing.assert_allclose(
        tn.to_numpy(nonorthogonal_frame.project(first).to_tt().full()),
        tn.to_numpy(projected_first.to_tt().full()),
        rtol=1e-9,
        atol=1e-9,
    )


def test_projection_transport_is_nonexpansive():
    base = _random_tt(d=4, n=3, r=2, seed=12)
    frame = TTManifoldFrame.from_tt(base)
    tangent = frame.random_tangent(seed=13)
    new_tensor = tangent.affine_to_tt(step=1e-2).round(eps=1e-14, rmax=2)
    target = TTManifoldFrame.from_tt(new_tensor)

    transported = projection_transport(tangent, target)
    assert float(tn.to_numpy(transported.norm()).item()) <= (
        float(tn.to_numpy(tangent.norm()).item()) + 1e-10
    )


def test_transported_factor_matches_dense_projection():
    base = _random_tt(d=3, n=3, r=2, seed=18)
    source = TTManifoldFrame.from_tt(base)
    columns = [source.random_tangent(seed=30 + k) for k in range(3)]
    batch = TTTangentBatch.from_columns(columns)
    target_tensor = source.retract(columns[0], step=1e-3)
    target = TTManifoldFrame.from_tt(target_tensor)
    transported = transport_batch(batch, target)

    target_basis = _dense_projector_basis(target)
    dense_source = np.stack(
        [tn.to_numpy(column.to_tt().full()).reshape(-1) for column in columns],
        axis=1,
    )
    dense_expected = target_basis @ (target_basis.T @ dense_source)
    dense_actual = np.stack(
        [
            tn.to_numpy(transported.column(k).to_tt().full()).reshape(-1)
            for k in range(transported.column_count)
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        dense_actual, dense_expected, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        tn.to_numpy(transported.gram()),
        dense_expected.T @ dense_expected,
        rtol=1e-9,
        atol=1e-9,
    )


def test_tangent_batch_orthonormalization_is_rank_revealing():
    frame = TTManifoldFrame.from_tt(_random_tt(d=3, n=3, r=2, seed=41))
    first = frame.random_tangent(seed=42)
    second = frame.random_tangent(seed=43)
    dependent = first.scaled(2.0).add(second.scaled(-0.5))
    batch = TTTangentBatch.from_columns([first, second, dependent])

    basis = batch.orthonormalize(relative_tolerance=1e-11)

    assert basis.column_count == 2
    np.testing.assert_allclose(
        tn.to_numpy(basis.gram()),
        np.eye(2),
        rtol=1e-9,
        atol=1e-9,
    )
    for column in [first, second, dependent]:
        coefficients = basis.adjoint_apply(column).reshape(-1, 1)
        reconstructed = basis.linear_combination(coefficients).column(0)
        error = reconstructed.add(column.scaled(-1.0))
        assert float(tn.to_numpy(error.norm()).item()) < 1e-9


def test_fixed_rank_retraction_has_first_order_accuracy():
    tensor = _random_tt(d=3, n=3, r=2, seed=16)
    frame = TTManifoldFrame.from_tt(tensor)
    tangent = frame.random_tangent(seed=17)
    dense_base = tn.to_numpy(tensor.full())
    dense_tangent = tn.to_numpy(tangent.to_tt().full())

    zero = frame.retract(tangent, step=0.0)
    np.testing.assert_allclose(
        tn.to_numpy(zero.full()), dense_base, rtol=1e-10, atol=1e-10
    )

    errors = []
    steps = [3.125e-3, 1.5625e-3, 7.8125e-4, 3.90625e-4]
    for step in steps:
        retracted = frame.retract(tangent, step=step)
        assert tuple(retracted.R) == frame.ranks
        error = np.linalg.norm(
            tn.to_numpy(retracted.full())
            - (dense_base + step * dense_tangent)
        )
        errors.append(error)

    scaled = np.asarray(errors) / np.square(steps)
    assert np.max(scaled) / np.min(scaled) < 1.1


def test_tangent_batch_gram_and_linear_combinations():
    frame = TTManifoldFrame.from_tt(_random_tt(d=3, n=3, r=2, seed=14))
    columns = [frame.random_tangent(seed=20 + k) for k in range(3)]
    batch = TTTangentBatch.from_columns(columns)

    dense_columns = np.stack(
        [tn.to_numpy(column.to_tt().full()).reshape(-1) for column in columns],
        axis=1,
    )
    np.testing.assert_allclose(
        tn.to_numpy(batch.gram()),
        dense_columns.T @ dense_columns,
        rtol=1e-10,
        atol=1e-10,
    )

    coefficients = np.array([[1.0, 0.5], [-0.25, 1.5], [0.75, -1.0]])
    combined = batch.linear_combination(coefficients)
    dense_combined = np.stack(
        [
            tn.to_numpy(combined.column(k).to_tt().full()).reshape(-1)
            for k in range(2)
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        dense_combined,
        dense_columns @ coefficients,
        rtol=1e-10,
        atol=1e-10,
    )

    selected = batch.select([2, 0]).scaled(0.25)
    dense_selected = np.stack(
        [
            tn.to_numpy(selected.column(k).to_tt().full()).reshape(-1)
            for k in range(2)
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        dense_selected,
        0.25 * dense_columns[:, [2, 0]],
        rtol=1e-10,
        atol=1e-10,
    )


def test_interface_singular_values_detect_rank_boundary():
    rng = np.random.default_rng(15)
    left, _ = np.linalg.qr(rng.standard_normal((4, 2)))
    right, _ = np.linalg.qr(rng.standard_normal((4, 2)))
    singular_values = np.array([1.0, 1e-9])
    cores = [
        tn.tensor(left.reshape(1, 4, 2), dtype=tn.float64),
        tn.tensor(
            (np.diag(singular_values) @ right.T).reshape(2, 4, 1),
            dtype=tn.float64,
        ),
    ]
    frame = TTManifoldFrame.from_tt(tt.TT(cores))
    measured = np.sort(
        tn.to_numpy(frame.interface_singular_values[0])
    )[::-1]
    np.testing.assert_allclose(
        measured, singular_values, rtol=1e-7, atol=1e-12
    )
    assert frame.regularity(tolerance=1e-8).regular is False
    assert frame.regularity(tolerance=1e-10).regular is True
