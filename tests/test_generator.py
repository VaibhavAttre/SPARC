import numpy as np
import pandas as pd
import pytest

from workloads.generators import (
    TRACE_COLUMNS,
    generate_all,
    generate_zipf,
    load_trace,
    save_traces,
)


def test_all_workloads_generate_valid_small_traces():
    traces = generate_all(num_requests=500, num_keys=50, seed=1)
    assert len(traces) == 7
    for name, t in traces.items():
        assert list(t.columns) == TRACE_COLUMNS, name
        assert len(t) == 500, name
        assert set(t["op"]) <= {"R", "W"}, name
        assert (np.diff(t["timestamp"]) > 0).all(), name


def test_same_seed_gives_same_trace():
    a = generate_zipf(num_requests=200, seed=3)
    b = generate_zipf(num_requests=200, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_small_trace_files_round_trip(tmp_path):
    traces = generate_all(num_requests=100, num_keys=20, seed=2, only=["zipf", "scan", "mixed"])
    save_traces(traces, tmp_path)
    for name, t in traces.items():
        pd.testing.assert_frame_equal(load_trace(tmp_path / f"{name}.csv"), t)


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        generate_zipf(write_fraction=2.0)