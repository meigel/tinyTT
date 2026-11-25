"""The benchmark harness runs, saves, compares and fails on a slowdown.

Timings themselves are not asserted -- they depend on the machine -- only that
the script produces them and that ``--fail-slower`` does its job, since a
harness that silently stopped working would look exactly like "no regression".
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _bench():
    spec = importlib.util.spec_from_file_location("bench", ROOT / "benchmarks" / "bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_harness_saves_every_case(tmp_path, capsys):
    bench = _bench()
    out = tmp_path / "baseline.json"
    assert bench.main(["--quick", "--repeat", "1", "--save", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert set(saved["timings"]) == set(bench._cases(quick=True))
    assert all(seconds > 0 for seconds in saved["timings"].values())
    assert saved["environment"]["quick"] is True
    assert "median" in capsys.readouterr().out


def test_compare_flags_a_case_slower_than_allowed(tmp_path, capsys):
    bench = _bench()
    baseline = tmp_path / "baseline.json"
    # a baseline that claims every case took a nanosecond: anything is slower
    baseline.write_text(json.dumps({"timings": {"round": 1e-9}}))
    code = bench.main(["--quick", "--repeat", "1", "--only", "round",
                       "--compare", str(baseline), "--fail-slower", "1.5"])
    assert code == 1
    assert "slower than allowed: round" in capsys.readouterr().out


def test_compare_passes_within_the_allowance(tmp_path):
    bench = _bench()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"timings": {"round": 1e9}}))
    assert bench.main(["--quick", "--repeat", "1", "--only", "round",
                       "--compare", str(baseline), "--fail-slower", "1.5"]) == 0
