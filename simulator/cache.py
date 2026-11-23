"""Minimal cache simulator: LRU eviction on top of libCacheSim's PluginCache.

libCacheSim owns capacity, hit/miss detection and eviction execution. We keep
the recency order in Python and hand it to the library through hooks.

Requests use the SPARC trace format (request_id, op, key, size, phase), either
as a dict (e.g. a csv.DictReader row or a pandas row) or any object with those
attributes. `phase` is ignored. Needs libcachesim (Linux/WSL/macOS only).
"""

from collections import OrderedDict

from libcachesim import PluginCache, ReqOp, Request

_OPS = {"R": ReqOp.OP_READ, "W": ReqOp.OP_WRITE}


def _field(req, name, default=None):
    if isinstance(req, dict):
        return req.get(name, default)
    return getattr(req, name, default)


class Cache:
    def __init__(self, capacity: int):
        """`capacity` is in size units; with size=1 requests it is an object count."""
        self.capacity = capacity
        self._lru = OrderedDict()  # least recently used first
        self.hits = 0
        self.misses = 0
        self._cache = PluginCache(
            cache_size=capacity,
            cache_init_hook=lambda params: self._lru,
            cache_hit_hook=self._on_hit,
            cache_miss_hook=self._on_miss,
            cache_eviction_hook=self._pick_victim,
            cache_remove_hook=lambda data, obj_id: None,
            cache_free_hook=lambda data: None,  # the binding rejects None
        )

    # --- libCacheSim hooks -------------------------------------------------
    def _on_hit(self, data, req):
        self._lru.move_to_end(req.obj_id)

    def _on_miss(self, data, req):  # fires when the object is inserted
        self._lru[req.obj_id] = None

    def _pick_victim(self, data, req):  # LRU: oldest entry
        victim, _ = self._lru.popitem(last=False)
        return victim

    # --- public API --------------------------------------------------------
    @property
    def requests(self) -> int:
        return self.hits + self.misses

    def keys(self) -> list:
        """Cached keys, least recently used first."""
        return list(self._lru)

    def access(self, req) -> bool:
        """Process one request. Returns True on a hit, False on a miss."""
        op = str(_field(req, "op", "R")).strip().upper()
        if op not in _OPS:
            raise ValueError(f"op must be 'R' or 'W', got {op!r}")
        lib_req = Request(
            obj_id=int(_field(req, "key")),
            obj_size=int(_field(req, "size", 1)),
            clock_time=int(_field(req, "request_id", self.requests)),
            op=_OPS[op],
        )
        hit = bool(self._cache.get(lib_req))
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        return hit
