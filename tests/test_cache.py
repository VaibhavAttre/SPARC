import pytest

pytest.importorskip("libcachesim")  # not installable on native Windows

from simulator.cache import Cache


def make_requests(keys):
    return [
        {"request_id": i, "op": "R", "key": k, "size": 1, "phase": "test"}
        for i, k in enumerate(keys)
    ]


def test_lru_hits_misses_and_eviction():
    cache = Cache(capacity=3)
    # Expected, capacity 3 (LRU order shown least -> most recent):
    #  1 miss [1]   2 miss [1,2]   3 miss [1,2,3]   1 hit [2,3,1]
    #  4 miss, evicts 2 [3,1,4]   1 hit [3,4,1]   2 miss, evicts 3 [4,1,2]
    expected = [False, False, False, True, False, True, False]

    results = [cache.access(r) for r in make_requests([1, 2, 3, 1, 4, 1, 2])]

    assert results == expected
    assert (cache.hits, cache.misses, cache.requests) == (2, 5, 7)
    assert cache.keys() == [4, 1, 2]  # 2 and 3 were each evicted once; 3 is gone


def test_state_persists_between_calls():
    cache = Cache(capacity=2)
    assert cache.access({"request_id": 0, "op": "R", "key": 7, "size": 1}) is False
    assert cache.access({"request_id": 1, "op": "W", "key": 7, "size": 1}) is True
