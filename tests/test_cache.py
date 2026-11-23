import pytest

pytest.importorskip("libcachesim")  # not installable on native Windows

from simulator.cache import Cache


def req(i, key, op="R"):
    return {"request_id": i, "op": op, "key": key, "size": 1, "phase": "test"}


def run(cache, keys, start=0):
    return [cache.access(req(start + i, k)) for i, k in enumerate(keys)]


# Capacity 3. Insert 1,2,3, then touch keys in this order. After it:
#   insertion (FIFO) order : 1, 2, 3    -> FIFO victim = 1
#   recency, oldest first  : 2, 3, 1    -> LRU victim  = 2
#   hit counts             : 1:5, 2:3, 3:2 -> LFU victim = 3
SETUP = [1, 2, 3, 2, 2, 3, 1, 1, 1, 1]


def test_lru_hits_misses_and_eviction():
    cache = Cache(capacity=3)
    # LRU order shown least -> most recent:
    #  1 miss [1]   2 miss [1,2]   3 miss [1,2,3]   1 hit [2,3,1]
    #  4 miss, evicts 2 [3,1,4]   1 hit [3,4,1]   2 miss, evicts 3 [4,1,2]
    results = run(cache, [1, 2, 3, 1, 4, 1, 2])

    assert results == [False, False, False, True, False, True, False]
    assert (cache.hits, cache.misses, cache.requests) == (2, 5, 7)
    assert cache.keys() == [4, 1, 2]


def test_state_persists_between_calls():
    cache = Cache(capacity=2)
    assert cache.access(req(0, 7)) is False
    assert cache.access(req(1, 7, "W")) is True


@pytest.mark.parametrize("policy, victim", [("LRU", 2), ("FIFO", 1), ("LFU", 3)])
def test_each_policy_evicts_its_own_victim(policy, victim):
    cache = Cache(capacity=3, replacement=policy)
    run(cache, SETUP)
    assert cache.access(req(100, 4)) is False
    assert sorted(cache.keys()) == sorted({1, 2, 3, 4} - {victim})


def test_switching_policy_midstream():
    cache = Cache(capacity=3, replacement="LRU")
    run(cache, SETUP)
    assert (cache.hits, cache.misses) == (7, 3)

    cache.set_policy("FIFO")  # oldest inserted is 1, even though it is the hottest key
    assert cache.access(req(100, 4)) is False
    assert sorted(cache.keys()) == [2, 3, 4]

    cache.set_policy("LFU")  # counts 2:3, 3:2, 4:1 -> evicts 4
    assert cache.access(req(101, 5)) is False
    assert sorted(cache.keys()) == [2, 3, 5]

    cache.set_policy("LRU")  # recency 2, 3, 5 -> evicts 2
    assert cache.access(req(102, 6)) is False
    assert sorted(cache.keys()) == [3, 5, 6]

    cache.set_policy("FIFO")  # insertion order 3, 5, 6 -> evicts 3
    assert cache.access(req(103, 7)) is False
    assert sorted(cache.keys()) == [5, 6, 7]

    # Contents and counters survived every switch.
    assert cache.access(req(104, 5)) is True
    assert (cache.hits, cache.misses, cache.requests) == (8, 7, 15)


def test_bad_policy_rejected():
    with pytest.raises(ValueError):
        Cache(capacity=2, replacement="MRU")
    with pytest.raises(ValueError):
        Cache(capacity=2).set_policy("ARC")


# --- write policy, dirty blocks, counters ------------------------------------
def acc(cache, ops_keys, start=0):
    """Run (op, key) pairs, e.g. [("W", 1), ("R", 2)]."""
    return [cache.access(req(start + i, k, o)) for i, (o, k) in enumerate(ops_keys)]


def test_write_back_dirty_eviction_counts_a_backing_write():
    cache = Cache(capacity=2, replacement="LRU", write_policy="WB")
    acc(cache, [("W", 1), ("R", 2)])
    s = cache.get_stats()
    assert (s["dirty_blocks"], s["backing_writes"]) == (1, 0)  # write deferred

    acc(cache, [("R", 3)], start=2)  # evicts LRU key 1, which is dirty
    s = cache.get_stats()
    assert s["dirty_evictions"] == 1 and s["backing_writes"] == 1
    assert s["dirty_blocks"] == 0
    assert (s["misses"], s["backing_reads"], s["write_requests"]) == (3, 3, 1)


def test_clean_eviction_costs_no_backing_write():
    cache = Cache(capacity=2, write_policy="WB")
    acc(cache, [("W", 1), ("R", 2), ("R", 3), ("R", 1)])
    # 3 evicts dirty 1 (1 write-back); re-reading 1 evicts clean 2 (no write).
    s = cache.get_stats()
    assert s["dirty_evictions"] == 1
    assert s["dirty_blocks"] == 0


def test_write_through_writes_every_write_and_keeps_nothing_dirty():
    cache = Cache(capacity=2, write_policy="WT")
    assert acc(cache, [("W", 1), ("W", 1)]) == [False, True]  # write miss, write hit
    s = cache.get_stats()
    assert (s["wt_writes"], s["backing_writes"], s["dirty_blocks"]) == (2, 2, 0)
    assert (s["backing_reads"], s["hits"]) == (1, 1)  # write-allocate read on the miss


def test_write_back_combines_repeated_writes():
    wt = Cache(capacity=2, write_policy="WT")
    wb = Cache(capacity=2, write_policy="WB")
    for c in (wt, wb):
        acc(c, [("W", 1)] * 3)
    assert wt.get_stats()["backing_writes"] == 3
    assert wb.get_stats()["backing_writes"] == 0
    assert wb.flush_all() == 1
    assert wb.get_stats()["backing_writes"] == 1  # 3 writes became 1


def test_flush_all():
    cache = Cache(capacity=3, write_policy="WB")
    acc(cache, [("W", 1), ("W", 2), ("R", 3), ("W", 1)])
    assert cache.get_stats()["dirty_blocks"] == 2
    assert cache.flush_all() == 2
    s = cache.get_stats()
    assert (s["dirty_blocks"], s["flush_writes"], s["backing_writes"]) == (0, 2, 2)
    assert cache.flush_all() == 0  # nothing left to flush
    assert sorted(cache.keys()) == [1, 2, 3]  # flushing leaves contents untouched


def test_switching_wb_to_wt_flushes_dirty_blocks():
    cache = Cache(capacity=3, write_policy="WB")
    acc(cache, [("W", 1), ("W", 2)])
    cache.set_policy(write_policy="WT")
    s = cache.get_stats()
    assert (s["dirty_blocks"], s["flush_writes"], s["switches"]) == (0, 2, 1)
    assert cache.policy == ("LRU", "WT")

    assert cache.access(req(10, 1, "W")) is True  # now write-through: immediate write
    s = cache.get_stats()
    assert (s["wt_writes"], s["backing_writes"], s["dirty_blocks"]) == (1, 3, 0)


def test_switching_wt_to_wb_needs_no_flush_and_counts_switches():
    cache = Cache(capacity=3, write_policy="WT")
    acc(cache, [("W", 1)])
    cache.set_policy("FIFO", "WB")
    assert cache.get_stats()["flush_writes"] == 0
    assert cache.policy == ("FIFO", "WB")
    cache.set_policy("FIFO", "WB")  # no change: no switch charged
    assert cache.get_stats()["switches"] == 1
    cache.set_policy("LFU")  # replacement only; write policy kept
    assert cache.policy == ("LFU", "WB")
    assert cache.get_stats()["switches"] == 2


def test_set_policy_is_atomic_on_bad_input():
    cache = Cache(capacity=2, write_policy="WB")
    with pytest.raises(ValueError):
        cache.set_policy("FIFO", "XX")
    assert cache.policy == ("LRU", "WB")
    assert cache.get_stats()["switches"] == 0


def test_stats_totals_are_consistent():
    cache = Cache(capacity=3, write_policy="WB")
    acc(cache, [("W", 1), ("R", 2), ("W", 3), ("R", 1), ("W", 4), ("R", 2)])
    cache.set_policy("LFU", "WT")
    acc(cache, [("W", 5), ("R", 5)], start=6)
    s = cache.get_stats()
    assert s["requests"] == s["hits"] + s["misses"] == 8
    assert s["read_requests"] + s["write_requests"] == 8
    assert s["backing_reads"] == s["misses"]
    assert s["backing_writes"] == s["wt_writes"] + s["dirty_evictions"] + s["flush_writes"]
    assert s["n_objects"] == 3
