from __future__ import annotations

import os

import numpy as np
import pytest

import tinytt._backend as tn
from tinytt.flow_matching import (
    build_velocity,
    domain_from_paths,
    evaluate_pairwise,
    make_banana_pair_data,
    make_four_mode_gaussian_pair_data,
    train_fm,
)


def _run_small_gate(source: np.ndarray, target: np.ndarray, *, seed: int) -> tuple[dict, dict]:
    d = source.shape[1]
    domain = domain_from_paths(source, target, pad_frac=0.08)
    vf = build_velocity(
        d,
        domain,
        poly_degree=4,
        time_degree=2,
        rank=None,
        init_scale=0.01,
        apply_cutoff=False,
        learnable_bias=True,
        seed=seed,
    )
    vf.output_bias.assign(tn.tensor((target - source).mean(axis=0), dtype=tn.float64))
    before = evaluate_pairwise(
        vf,
        source,
        target,
        n_eval=128,
        n_steps=20,
        method="euler",
        vmax=5.0,
        include_sinkhorn=True,
    )
    train_fm(
        vf,
        source,
        target,
        epochs=180,
        batch_size=128,
        lr=1e-2,
        seed=seed + 17,
        grad_clip_norm=1.0,
        paired=True,
    )
    after = evaluate_pairwise(
        vf,
        source,
        target,
        n_eval=128,
        n_steps=20,
        method="euler",
        vmax=5.0,
        include_sinkhorn=True,
    )
    return before, after


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_TINYTT_FM_BENCHMARK") != "1",
    reason="set RUN_TINYTT_FM_BENCHMARK=1 to run the tinyTT FM convergence gate",
)
@pytest.mark.parametrize("case", ["banana", "gm"])
def test_d10_flow_matching_reduces_energy_and_sinkhorn(case):
    if case == "banana":
        source, target = make_banana_pair_data(512, 10, curvature=1.5, angle_deg=45.0, seed=0)
    else:
        source, target = make_four_mode_gaussian_pair_data(512, 10, seed=0)
    before, after = _run_small_gate(source, target, seed=0)
    assert after["energy"] < 0.35 * before["energy"]
    assert after["sinkhorn"] < 0.35 * before["sinkhorn"]
