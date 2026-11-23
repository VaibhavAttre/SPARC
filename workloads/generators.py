"""Synthetic workload generators for the cache-policy RL project.

Every generator returns a pandas DataFrame with the request schema from the
README plus a timestamp column:  request_id, timestamp, op, key, size, phase.

Workloads (each can be called on its own, or all together via generate_all):

    generate_zipf              Zipfian popularity
    generate_temporal          Temporal locality (recent keys get re-accessed)
    generate_scan              Sequential scan (distinct keys, once each)
    generate_hot_shift         Hot set changes partway through the trace
    generate_write_ramp        Read-heavy -> write-heavy (write fraction ramps)
    generate_write_dispersion  Repeated writes -> dispersed writes
    generate_mixed             Locality, popularity and R/W mix change together

Design notes
------------
* Zipf popularity is sampled from a *finite* key universe with
  P(rank r) proportional to 1 / r**alpha. numpy.random.zipf is not used because
  it needs alpha > 1 and has unbounded support.
* Popularity rank is mapped to key IDs through a seeded random permutation, so
  "hot" keys are not simply the lowest IDs, and a different permutation gives
  a different hot set (used by the hot-set shift workload).
* All randomness flows from a single seed, so traces are fully reproducible.
* The `phase` column is a label for analysis only. Do not feed it to the agent.
* `timestamp` is simulated arrival time in seconds (float, non-decreasing).
  Gaps are exponential (Poisson arrivals) by default, or fixed with
  arrival="constant"; `arrival_rate` is requests per simulated second.
  Timestamps are drawn last, so they never change which keys/ops a seed makes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TRACE_COLUMNS = ["request_id", "timestamp", "op", "key", "size", "phase"]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _check_common(num_requests, num_keys, write_fraction, block_size) -> None:
    if num_requests < 0:
        raise ValueError("num_requests must be >= 0")
    if num_keys < 1:
        raise ValueError("num_keys must be >= 1")
    if not 0.0 <= write_fraction <= 1.0:
        raise ValueError("write_fraction must be in [0, 1]")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")


def _frame(keys, ops, timestamps, phase, start_id: int, block_size: int) -> pd.DataFrame:
    """Assemble a trace DataFrame. `phase` may be a string or a per-row array."""
    n = len(keys)
    return pd.DataFrame(
        {
            "request_id": np.arange(start_id, start_id + n, dtype=np.int64),
            "timestamp": np.asarray(timestamps, dtype=np.float64),
            "op": ops,
            "key": np.asarray(keys, dtype=np.int64),
            "size": np.full(n, block_size, dtype=np.int64),
            "phase": phase,
        },
        columns=TRACE_COLUMNS,
    )


def _random_ops(rng: np.random.Generator, n: int, write_fraction) -> np.ndarray:
    """'W' with probability write_fraction (scalar or per-request array)."""
    return np.where(rng.random(n) < write_fraction, "W", "R")


_ARRIVALS = ("poisson", "constant")


def _check_arrival(arrival_rate, arrival, start_time) -> None:
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be > 0")
    if arrival not in _ARRIVALS:
        raise ValueError(f"arrival must be one of {_ARRIVALS}")
    if start_time < 0:
        raise ValueError("start_time must be >= 0")


def _timestamps(rng, n: int, arrival_rate: float, arrival: str, start_time: float) -> np.ndarray:
    """Simulated arrival times in seconds: start_time + cumulative gaps.

    `start_time` is the clock just before the first request, so the first
    timestamp is start_time plus one gap. Draw this *after* keys and ops so the
    random stream for those is unaffected.
    """
    if arrival == "constant":
        gaps = np.full(n, 1.0 / arrival_rate)
    else:
        gaps = rng.exponential(1.0 / arrival_rate, size=n)
    return start_time + np.cumsum(gaps)


def _spawn_seeds(seed: int, n: int) -> list[int]:
    """n independent integer seeds derived from one base seed."""
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]


def _segment_bounds(n: int, points: Sequence[float]) -> list[tuple[int, int]]:
    """Split range(n) at fractional points into (start, end) segments."""
    pts = list(points)
    if any(not 0.0 < p < 1.0 for p in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("shift_points must be strictly increasing values in (0, 1)")
    edges = [0] + [int(round(p * n)) for p in pts] + [n]
    return list(zip(edges[:-1], edges[1:]))


# --------------------------------------------------------------------------- #
# 1. Zipfian popularity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ZipfConfig:
    """Parameters for a Zipfian trace."""

    num_requests: int = 100_000
    num_keys: int = 10_000          # size of the key universe
    alpha: float = 1.0              # skew: 0 = uniform, larger = more skewed
    write_fraction: float = 0.2     # P(op == "W"), independent of the key
    seed: int = 0
    key_offset: int = 0             # shift key IDs (lets phases use disjoint keys)
    permute_keys: bool = True       # decouple popularity rank from key ID
    block_size: int = 1             # fixed-size blocks for the MVP
    phase: str = "zipf"             # label for analysis only, not for the agent
    start_id: int = 0               # first request_id (for concatenating phases)
    arrival_rate: float = 1000.0    # requests per simulated second
    arrival: str = "poisson"        # "poisson" (random gaps) or "constant"
    start_time: float = 0.0         # simulated clock before the first request

    def validate(self) -> None:
        _check_common(self.num_requests, self.num_keys, self.write_fraction, self.block_size)
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        _check_arrival(self.arrival_rate, self.arrival, self.start_time)


def zipf_probabilities(num_keys: int, alpha: float) -> np.ndarray:
    """Normalized Zipf probabilities for ranks 1..num_keys."""
    ranks = np.arange(1, num_keys + 1, dtype=np.float64)
    weights = ranks ** (-alpha)
    return weights / weights.sum()


def _zipf_keys(rng, n: int, num_keys: int, alpha: float, permute: bool = True) -> np.ndarray:
    rank_to_key = rng.permutation(num_keys) if permute else np.arange(num_keys)
    ranks = rng.choice(num_keys, size=n, p=zipf_probabilities(num_keys, alpha))
    return rank_to_key[ranks]


def generate_zipf(config: ZipfConfig | None = None, **overrides) -> pd.DataFrame:
    """Zipfian popularity: a few keys receive most of the accesses.

    Usage:
        trace = generate_zipf(num_requests=50_000, alpha=1.1, seed=3)
        trace = generate_zipf(ZipfConfig(alpha=0.8), seed=7)   # overrides win
    """
    cfg = config or ZipfConfig()
    if overrides:
        cfg = ZipfConfig(**{**asdict(cfg), **overrides})
    cfg.validate()

    rng = np.random.default_rng(cfg.seed)
    keys = _zipf_keys(rng, cfg.num_requests, cfg.num_keys, cfg.alpha, cfg.permute_keys)
    keys = keys + cfg.key_offset
    ops = _random_ops(rng, cfg.num_requests, cfg.write_fraction)
    ts = _timestamps(rng, cfg.num_requests, cfg.arrival_rate, cfg.arrival, cfg.start_time)
    return _frame(keys, ops, ts, cfg.phase, cfg.start_id, cfg.block_size)


# --------------------------------------------------------------------------- #
# 2. Temporal locality
# --------------------------------------------------------------------------- #
def generate_temporal(
    num_requests: int = 100_000,
    num_keys: int = 10_000,
    reuse_prob: float = 0.8,
    mean_reuse_distance: float = 20.0,
    write_fraction: float = 0.2,
    seed: int = 0,
    key_offset: int = 0,
    block_size: int = 1,
    phase: str = "temporal",
    start_id: int = 0,
    arrival_rate: float = 1000.0,
    arrival: str = "poisson",
    start_time: float = 0.0,
) -> pd.DataFrame:
    """Temporal locality: recently accessed keys are likely to be accessed again.

    With probability `reuse_prob`, request i repeats the key requested d steps
    earlier, where d is geometric with mean `mean_reuse_distance` (measured in
    requests). Otherwise it draws a fresh key uniformly from the universe, so
    the working set drifts over time. Small mean distances favor LRU; popularity
    is deliberately not skewed, so LFU has no frequency signal to exploit.
    """
    _check_common(num_requests, num_keys, write_fraction, block_size)
    _check_arrival(arrival_rate, arrival, start_time)
    if not 0.0 <= reuse_prob <= 1.0:
        raise ValueError("reuse_prob must be in [0, 1]")
    if mean_reuse_distance < 1.0:
        raise ValueError("mean_reuse_distance must be >= 1")

    rng = np.random.default_rng(seed)
    n = num_requests
    keys = rng.integers(0, num_keys, size=n).tolist()            # fresh-key fallback
    reuse = (rng.random(n) < reuse_prob).tolist()
    dist = rng.geometric(1.0 / mean_reuse_distance, size=n).tolist()  # >= 1
    for i in range(1, n):
        if reuse[i]:
            keys[i] = keys[i - min(dist[i], i)]

    keys = np.asarray(keys, dtype=np.int64) + key_offset
    ops = _random_ops(rng, n, write_fraction)
    ts = _timestamps(rng, n, arrival_rate, arrival, start_time)
    return _frame(keys, ops, ts, phase, start_id, block_size)


# --------------------------------------------------------------------------- #
# 3. Sequential scan
# --------------------------------------------------------------------------- #
def generate_scan(
    num_requests: int = 100_000,
    num_keys: int | None = None,
    write_fraction: float = 0.1,
    start_key: int = 0,
    seed: int = 0,
    key_offset: int = 0,
    block_size: int = 1,
    phase: str = "scan",
    start_id: int = 0,
    arrival_rate: float = 1000.0,
    arrival: str = "poisson",
    start_time: float = 0.0,
) -> pd.DataFrame:
    """Sequential scan: keys are accessed in order, each distinct.

    `num_keys=None` makes the scan length equal to num_requests, so every
    request touches a never-before-seen key (a pure one-pass scan). If
    num_keys < num_requests the scan wraps around and becomes a cyclic loop,
    which is the classic worst case for LRU/FIFO when num_keys is slightly
    larger than the cache capacity.
    """
    span = max(1, num_requests if num_keys is None else num_keys)
    _check_common(num_requests, span, write_fraction, block_size)
    _check_arrival(arrival_rate, arrival, start_time)

    rng = np.random.default_rng(seed)
    keys = (start_key + np.arange(num_requests, dtype=np.int64)) % span + key_offset
    ops = _random_ops(rng, num_requests, write_fraction)
    ts = _timestamps(rng, num_requests, arrival_rate, arrival, start_time)
    return _frame(keys, ops, ts, phase, start_id, block_size)


# --------------------------------------------------------------------------- #
# 4. Hot-set shift
# --------------------------------------------------------------------------- #
def generate_hot_shift(
    num_requests: int = 100_000,
    num_keys: int = 10_000,
    alpha: float = 1.0,
    write_fraction: float = 0.2,
    shift_points: Sequence[float] = (0.5,),
    recurring: bool = False,
    seed: int = 0,
    key_offset: int = 0,
    block_size: int = 1,
    start_id: int = 0,
    arrival_rate: float = 1000.0,
    arrival: str = "poisson",
    start_time: float = 0.0,
) -> pd.DataFrame:
    """Hot-set shift: the popular keys change partway through the trace.

    Each segment is Zipf over the same key universe, but with its own
    popularity-to-key mapping, so keys that were hot become cold (this is what
    traps LFU's stale counts). `shift_points` are fractions of the trace where
    the hot set changes. With recurring=True the trace alternates between two
    hot sets (A, B, A, B, ...) so you can test whether an agent recognizes a
    phase it has seen before.

    Phase labels: hot-set-A, hot-set-B, ...
    """
    _check_common(num_requests, num_keys, write_fraction, block_size)
    if alpha < 0:
        raise ValueError("alpha must be >= 0")

    _check_arrival(arrival_rate, arrival, start_time)

    rng = np.random.default_rng(seed)
    bounds = _segment_bounds(num_requests, shift_points)
    n_sets = 2 if recurring else len(bounds)
    perms = [rng.permutation(num_keys) for _ in range(n_sets)]
    probs = zipf_probabilities(num_keys, alpha)

    keys = np.empty(num_requests, dtype=np.int64)
    phase = np.empty(num_requests, dtype=object)
    for i, (start, end) in enumerate(bounds):
        which = i % 2 if recurring else i
        keys[start:end] = perms[which][rng.choice(num_keys, size=end - start, p=probs)]
        phase[start:end] = f"hot-set-{chr(ord('A') + which % 26)}"

    ops = _random_ops(rng, num_requests, write_fraction)
    ts = _timestamps(rng, num_requests, arrival_rate, arrival, start_time)
    return _frame(keys + key_offset, ops, ts, phase, start_id, block_size)


# --------------------------------------------------------------------------- #
# 5. Read-heavy -> write-heavy
# --------------------------------------------------------------------------- #
def generate_write_ramp(
    num_requests: int = 100_000,
    num_keys: int = 10_000,
    alpha: float = 1.0,
    start_write_fraction: float = 0.05,
    end_write_fraction: float = 0.8,
    shape: str = "linear",
    step_at: float = 0.5,
    seed: int = 0,
    key_offset: int = 0,
    block_size: int = 1,
    start_id: int = 0,
    arrival_rate: float = 1000.0,
    arrival: str = "poisson",
    start_time: float = 0.0,
) -> pd.DataFrame:
    """Read-heavy -> write-heavy: the write fraction changes over time.

    Key popularity is a fixed Zipf distribution, so only the R/W mix varies.
    shape="linear" ramps the write probability smoothly from start to end;
    shape="step" jumps from start to end at `step_at`. Swap start and end to
    get a write-heavy -> read-heavy trace.

    Phase labels: "read-heavy" while the write probability is below the
    midpoint of start/end, "write-heavy" afterwards (analysis only).
    """
    _check_common(num_requests, num_keys, start_write_fraction, block_size)
    if not 0.0 <= end_write_fraction <= 1.0:
        raise ValueError("end_write_fraction must be in [0, 1]")
    _check_arrival(arrival_rate, arrival, start_time)
    if shape not in ("linear", "step"):
        raise ValueError("shape must be 'linear' or 'step'")
    if shape == "step" and not 0.0 < step_at < 1.0:
        raise ValueError("step_at must be in (0, 1)")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")

    rng = np.random.default_rng(seed)
    n = num_requests
    keys = _zipf_keys(rng, n, num_keys, alpha) + key_offset

    if shape == "linear":
        p = np.linspace(start_write_fraction, end_write_fraction, n)
    else:
        p = np.where(np.arange(n) < round(step_at * n), start_write_fraction, end_write_fraction)

    ops = _random_ops(rng, n, p)
    if start_write_fraction == end_write_fraction:
        midpoint = 0.5
    else:
        midpoint = (start_write_fraction + end_write_fraction) / 2.0
    phase = np.where(p < midpoint, "read-heavy", "write-heavy")
    ts = _timestamps(rng, n, arrival_rate, arrival, start_time)
    return _frame(keys, ops, ts, phase, start_id, block_size)


# --------------------------------------------------------------------------- #
# 6. Repeated writes -> dispersed writes
# --------------------------------------------------------------------------- #
def generate_write_dispersion(
    num_requests: int = 100_000,
    num_keys: int = 10_000,
    alpha: float = 1.0,
    write_fraction: float = 0.4,
    repeated_write_keys: int = 16,
    dispersed_write_keys: int | None = None,
    switch_at: float = 0.5,
    order: str = "repeated_first",
    seed: int = 0,
    key_offset: int = 0,
    block_size: int = 1,
    start_id: int = 0,
    arrival_rate: float = 1000.0,
    arrival: str = "poisson",
    start_time: float = 0.0,
) -> pd.DataFrame:
    """Repeated writes -> dispersed writes.

    Reads follow a fixed Zipf distribution throughout. Writes (at a constant
    `write_fraction`) behave in two ways:
      * repeated:  uniformly over a small set of `repeated_write_keys` blocks,
                   so write-back can coalesce many writes into few flushes;
      * dispersed: uniformly over `dispersed_write_keys` blocks (default: the
                   whole universe), so nearly every write dirties a different
                   block and write-back pays for dirty evictions.
    order="repeated_first" or "dispersed_first" picks which comes first;
    `switch_at` is the fraction of the trace where the behavior changes.

    Phase labels: "repeated-writes" and "dispersed-writes" (applied to every
    request in that part of the trace, reads included).
    """
    _check_common(num_requests, num_keys, write_fraction, block_size)
    _check_arrival(arrival_rate, arrival, start_time)
    if not 0.0 < switch_at < 1.0:
        raise ValueError("switch_at must be in (0, 1)")
    if order not in ("repeated_first", "dispersed_first"):
        raise ValueError("order must be 'repeated_first' or 'dispersed_first'")
    if repeated_write_keys < 1:
        raise ValueError("repeated_write_keys must be >= 1")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    n_repeated = min(repeated_write_keys, num_keys)
    n_dispersed = num_keys if dispersed_write_keys is None else min(dispersed_write_keys, num_keys)
    if n_dispersed < 1:
        raise ValueError("dispersed_write_keys must be >= 1")

    rng = np.random.default_rng(seed)
    n = num_requests
    read_keys = _zipf_keys(rng, n, num_keys, alpha)
    is_write = rng.random(n) < write_fraction

    repeated_set = rng.choice(num_keys, size=n_repeated, replace=False)
    dispersed_set = rng.permutation(num_keys)[:n_dispersed]
    repeated_keys = repeated_set[rng.integers(0, n_repeated, size=n)]
    dispersed_keys = dispersed_set[rng.integers(0, n_dispersed, size=n)]

    in_first = np.arange(n) < round(switch_at * n)
    use_repeated = in_first == (order == "repeated_first")
    write_keys = np.where(use_repeated, repeated_keys, dispersed_keys)

    keys = np.where(is_write, write_keys, read_keys) + key_offset
    ops = np.where(is_write, "W", "R")
    phase = np.where(use_repeated, "repeated-writes", "dispersed-writes")
    ts = _timestamps(rng, n, arrival_rate, arrival, start_time)
    return _frame(keys, ops, ts, phase, start_id, block_size)


# --------------------------------------------------------------------------- #
# 7. Mixed phases
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhaseSpec:
    """One segment of a mixed trace.

    kind            "zipf", "temporal" or "scan"
    fraction        share of the trace (specs are normalized to sum to 1)
    write_fraction  R/W mix inside this segment
    label           phase label (defaults to `kind`)
    params          extra keyword args for that generator (e.g. {"alpha": 1.2})
    """

    kind: str
    fraction: float
    write_fraction: float
    label: str | None = None
    params: dict = field(default_factory=dict)


def default_mixed_schedule() -> list[PhaseSpec]:
    """Locality, popularity and R/W mix all change from phase to phase."""
    return [
        PhaseSpec("zipf", 0.25, 0.05, "zipf-read-heavy", {"alpha": 1.2}),
        PhaseSpec("temporal", 0.20, 0.30, "temporal-mixed",
                  {"reuse_prob": 0.85, "mean_reuse_distance": 15.0}),
        PhaseSpec("scan", 0.10, 0.10, "scan"),
        PhaseSpec("zipf", 0.25, 0.60, "zipf-write-heavy", {"alpha": 0.8}),
        PhaseSpec("temporal", 0.20, 0.50, "temporal-write-heavy",
                  {"reuse_prob": 0.7, "mean_reuse_distance": 40.0}),
    ]


def generate_mixed(
    num_requests: int = 100_000,
    num_keys: int = 10_000,
    phases: Iterable[PhaseSpec] | None = None,
    seed: int = 0,
    key_offset: int = 0,
    block_size: int = 1,
    start_id: int = 0,
    arrival_rate: float = 1000.0,
    arrival: str = "poisson",
    start_time: float = 0.0,
) -> pd.DataFrame:
    """Mixed phases: concatenated segments that change locality, popularity and
    read/write mix together, over one shared key universe (so the cache is not
    reset between phases). Each Zipf segment gets its own seed, and therefore
    its own hot set.

    Pass `phases=[PhaseSpec(...), ...]` for a custom schedule. A scan segment
    wraps if its length exceeds num_keys. Timestamps continue across phases; give a
    phase its own rate with PhaseSpec(..., params={"arrival_rate": 50.0}).
    """
    _check_common(num_requests, num_keys, 0.0, block_size)
    _check_arrival(arrival_rate, arrival, start_time)
    specs = list(phases) if phases is not None else default_mixed_schedule()
    if not specs:
        raise ValueError("phases must not be empty")
    for spec in specs:
        if spec.kind not in _KIND_GENERATORS:
            raise ValueError(f"unknown phase kind {spec.kind!r}; use {sorted(_KIND_GENERATORS)}")
        if spec.fraction <= 0:
            raise ValueError("PhaseSpec.fraction must be > 0")

    fractions = np.array([s.fraction for s in specs], dtype=np.float64)
    edges = np.rint(np.cumsum(fractions) / fractions.sum() * num_requests).astype(np.int64)
    edges = np.concatenate([[0], edges])
    edges[-1] = num_requests
    counts = np.diff(edges)

    frames = []
    clock = start_time
    for spec, count, start, sub_seed in zip(
        specs, counts, edges[:-1], _spawn_seeds(seed, len(specs))
    ):
        if count == 0:
            continue
        kwargs = dict(
            num_requests=int(count),
            num_keys=num_keys,
            write_fraction=spec.write_fraction,
            seed=sub_seed,
            key_offset=key_offset,
            block_size=block_size,
            phase=spec.label or spec.kind,
            start_id=start_id + int(start),
            arrival_rate=arrival_rate,
            arrival=arrival,
            start_time=clock,
        )
        kwargs.update(spec.params)
        frame = _KIND_GENERATORS[spec.kind](**kwargs)
        frames.append(frame)
        clock = float(frame["timestamp"].iloc[-1])

    if not frames:
        return _frame(np.empty(0, dtype=np.int64), np.empty(0, dtype=str), np.empty(0),
                      "mixed", start_id, block_size)
    return pd.concat(frames, ignore_index=True)


_KIND_GENERATORS = {
    "zipf": generate_zipf,
    "temporal": generate_temporal,
    "scan": generate_scan,
}


# --------------------------------------------------------------------------- #
# All workloads together
# --------------------------------------------------------------------------- #
WORKLOADS = {
    "zipf": generate_zipf,
    "temporal": generate_temporal,
    "scan": generate_scan,
    "hot_shift": generate_hot_shift,
    "write_ramp": generate_write_ramp,
    "write_dispersion": generate_write_dispersion,
    "mixed": generate_mixed,
}


def generate_all(
    num_requests: int = 100_000,
    num_keys: int = 10_000,
    seed: int = 0,
    block_size: int = 1,
    arrival_rate: float = 1000.0,
    arrival: str = "poisson",
    only: Iterable[str] | None = None,
    overrides: dict[str, dict] | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate several workloads at once. Returns {name: trace DataFrame}.

    * Each workload gets its own seed derived from `seed`, so traces are not
      correlated, and a workload's trace is the same whether you ask for all of
      them or just that one via `only`.
    * `num_keys` is shared, except for the pure scan, whose length defaults to
      num_requests (every key distinct). Override it with overrides={"scan": {"num_keys": ...}}.
    * `overrides` maps a workload name to extra keyword arguments, e.g.
      overrides={"zipf": {"alpha": 1.3}, "hot_shift": {"shift_points": (0.3, 0.6)}}.
    """
    overrides = overrides or {}
    names = list(WORKLOADS) if only is None else list(only)
    unknown = [n for n in list(names) + list(overrides) if n not in WORKLOADS]
    if unknown:
        raise ValueError(f"unknown workload(s) {unknown}; choose from {list(WORKLOADS)}")

    seeds = dict(zip(WORKLOADS, _spawn_seeds(seed, len(WORKLOADS))))
    traces = {}
    for name in names:
        kwargs = dict(
            num_requests=num_requests, num_keys=num_keys, seed=seeds[name],
            block_size=block_size, arrival_rate=arrival_rate, arrival=arrival,
        )
        if name == "scan":
            del kwargs["num_keys"]
        kwargs.update(overrides.get(name, {}))
        traces[name] = WORKLOADS[name](**kwargs)
    return traces


# --------------------------------------------------------------------------- #
# I/O and utilities
# --------------------------------------------------------------------------- #
def save_trace(trace: pd.DataFrame, path: str | Path) -> None:
    """Write a trace to CSV using the README schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(path, index=False)


def save_traces(traces: dict[str, pd.DataFrame], directory: str | Path) -> list[Path]:
    """Write each trace to <directory>/<name>.csv and return the paths."""
    directory = Path(directory)
    paths = []
    for name, trace in traces.items():
        path = directory / f"{name}.csv"
        save_trace(trace, path)
        paths.append(path)
    return paths


def load_trace(path: str | Path) -> pd.DataFrame:
    """Read a trace CSV and check that it matches the schema."""
    trace = pd.read_csv(path)
    missing = [c for c in TRACE_COLUMNS if c not in trace.columns]
    if missing:
        raise ValueError(f"trace is missing columns: {missing}")
    if not trace["op"].isin(["R", "W"]).all():
        raise ValueError("op column must contain only 'R' or 'W'")
    if not trace["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp column must be non-decreasing")
    return trace[TRACE_COLUMNS]


def split_seeds(base_seed: int = 0) -> dict[str, int]:
    """Distinct seed ranges so train / val / test never share a trace.

    Use as:  seed = split_seeds()["train"] + i   for i in range(n_traces)
    """
    return {
        "train": base_seed + 0,
        "val": base_seed + 1_000_000,
        "test": base_seed + 2_000_000,
    }


def summarize(trace: pd.DataFrame) -> dict:
    """Quick descriptive stats for sanity-checking a trace."""
    if len(trace) == 0:
        return {"requests": 0, "unique_keys": 0, "write_frac": 0.0, "top1%_share": 0.0,
                "duration_s": 0.0, "phases": ""}
    counts = trace["key"].value_counts()
    top = max(1, int(0.01 * len(counts)))
    return {
        "requests": len(trace),
        "unique_keys": int(len(counts)),
        "write_frac": round(float((trace["op"] == "W").mean()), 3),
        "top1%_share": round(float(counts.head(top).sum() / len(trace)), 3),
        "duration_s": round(float(trace["timestamp"].iloc[-1]), 3),
        "phases": " > ".join(trace["phase"].drop_duplicates()),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic cache workloads.")
    parser.add_argument("--num-requests", type=int, default=100_000)
    parser.add_argument("--num-keys", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="+", choices=list(WORKLOADS), help="subset of workloads")
    parser.add_argument("--out", default=Path("data"))
    args = parser.parse_args()

    traces = generate_all(args.num_requests, args.num_keys, args.seed, only=args.only)
    table = pd.DataFrame({name: summarize(t) for name, t in traces.items()}).T
    print(table.to_string())
    if args.out:
        paths = save_traces(traces, args.out)
        print(f"\nWrote {len(paths)} trace(s) to {args.out}")


if __name__ == "__main__":
    _main()