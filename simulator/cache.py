"""Cache simulator: switchable LRU/FIFO/LFU replacement and WT/WB write policy,
on top of libCacheSim's PluginCache.

libCacheSim owns capacity, hit/miss detection and eviction execution. We keep
metadata for all three replacement policies at once (recency order, insertion
order, hit counts), so `set_policy` can change the rule at any time without
touching the cache contents. Dirty blocks (write-back) are tracked in a set.

Write semantics (README MVP assumptions):
* Write-allocate: a write miss loads the block (one backing read), then the
  write is applied under the active write policy.
* Write-through (WT): every write request is also written to the backing store.
* Write-back (WB): a write only marks the block dirty. The backing write happens
  when the dirty block is evicted or flushed.
* Switching WB -> WT flushes every dirty block first (counted as flush writes).

Requests use the SPARC trace format (request_id, op, key, size, phase), either
as a dict (e.g. a csv.DictReader row) or any object with those attributes.
`phase` and `timestamp` are ignored. `size` is in logical block units (the MVP
uses 1, so capacity is a block count). Needs libcachesim (Linux/WSL/macOS only).
"""

from collections import OrderedDict

from libcachesim import PluginCache, ReqOp, Request

REPLACEMENT_POLICIES = ("LRU", "FIFO", "LFU")
WRITE_POLICIES = ("WT", "WB")
_OPS = {"R": ReqOp.OP_READ, "W": ReqOp.OP_WRITE}


def _field(req, name, default=None):
    if isinstance(req, dict):
        return req.get(name, default)
    return getattr(req, name, default)


def _check(value, allowed, what) -> str:
    value = str(value).upper()
    if value not in allowed:
        raise ValueError(f"{what} must be one of {allowed}, got {value!r}")
    return value


class Cache:
    def __init__(self, capacity: int, replacement: str = "LRU", write_policy: str = "WT"):
        """`capacity` is in size units; with size=1 requests it is a block count."""
        self.capacity = capacity
        self._replacement = _check(replacement, REPLACEMENT_POLICIES, "replacement")
        self._write_policy = _check(write_policy, WRITE_POLICIES, "write_policy")

        # Metadata for every replacement policy, kept current whichever is active.
        self._recency = OrderedDict()  # least recently used first (LRU)
        self._arrival = OrderedDict()  # insertion order, never reordered (FIFO)
        self._freq = {}                # hits+1 per key (LFU)
        self._dirty = set()            # keys modified but not yet written back

        # Cumulative event counters (the environment diffs these per window).
        self.hits = 0
        self.misses = 0
        self.read_requests = 0
        self.write_requests = 0
        self.backing_reads = 0     # every miss loads the block (incl. write-allocate)
        self.wt_writes = 0         # write-through: one backing write per write request
        self.dirty_evictions = 0   # write-back: dirty victim written on eviction
        self.flush_writes = 0      # dirty blocks written by flush_all / WB->WT switch
        self.switches = 0          # set_policy calls that changed the pair

        self._cache = PluginCache(
            cache_size=capacity,
            cache_init_hook=lambda _params: None,
            cache_hit_hook=self._on_hit,
            cache_miss_hook=self._on_miss,
            cache_eviction_hook=self._pick_victim,
            cache_remove_hook=lambda _data, _obj_id: None,
            cache_free_hook=lambda _data: None,  # the binding rejects None
        )

    # --- policy ------------------------------------------------------------
    @property
    def replacement(self) -> str:
        return self._replacement

    @property
    def write_policy(self) -> str:
        return self._write_policy

    @property
    def policy(self) -> tuple:
        return (self._replacement, self._write_policy)

    def set_policy(self, replacement: str = None, write_policy: str = None) -> None:
        """Switch policies without clearing the cache. `None` keeps the current one.

        Both arguments are validated before anything changes. Going WB -> WT
        flushes all dirty blocks first. A call that changes the pair counts as
        one switch.
        """
        new_repl = self._replacement if replacement is None else _check(
            replacement, REPLACEMENT_POLICIES, "replacement")
        new_write = self._write_policy if write_policy is None else _check(
            write_policy, WRITE_POLICIES, "write_policy")
        if (new_repl, new_write) == self.policy:
            return
        if self._write_policy == "WB" and new_write == "WT":
            self.flush_all()
        self._replacement, self._write_policy = new_repl, new_write
        self.switches += 1

    # --- libCacheSim hooks -------------------------------------------------
    def _on_hit(self, _data, req):
        self._recency.move_to_end(req.obj_id)
        self._freq[req.obj_id] += 1

    def _on_miss(self, _data, req):  # fires when the object is inserted
        key = req.obj_id
        self._recency[key] = None
        self._arrival[key] = None
        self._freq[key] = 1

    def _pick_victim(self, _data, _req):
        if self._replacement == "LRU":
            victim = next(iter(self._recency))
        elif self._replacement == "FIFO":
            victim = next(iter(self._arrival))
        else:  # LFU: lowest count; min() takes the first minimum, i.e. the LRU one
            victim = min(self._recency, key=self._freq.__getitem__)
        del self._recency[victim], self._arrival[victim], self._freq[victim]
        if victim in self._dirty:
            self._dirty.discard(victim)
            self.dirty_evictions += 1
        return victim

    # --- public API --------------------------------------------------------
    @property
    def requests(self) -> int:
        return self.hits + self.misses

    def keys(self) -> list:
        """Cached keys, least recently used first."""
        return list(self._recency)

    def access(self, req) -> bool:
        """Process one request. Returns True on a hit, False on a miss."""
        op = str(_field(req, "op", "R")).strip().upper()
        if op not in _OPS:
            raise ValueError(f"op must be 'R' or 'W', got {op!r}")
        key = int(_field(req, "key"))
        lib_req = Request(
            obj_id=key,
            obj_size=int(_field(req, "size", 1)),
            clock_time=int(_field(req, "request_id", self.requests)),
            op=_OPS[op],
        )
        hit = bool(self._cache.get(lib_req))  # may evict; the block is cached after
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            self.backing_reads += 1

        if op == "R":
            self.read_requests += 1
        else:
            self.write_requests += 1
            if self._write_policy == "WB":
                self._dirty.add(key)
            else:
                self.wt_writes += 1
        return hit

    def flush_all(self) -> int:
        """Write every dirty block to the backing store. Returns how many."""
        n = len(self._dirty)
        self.flush_writes += n
        self._dirty.clear()
        return n

    def get_stats(self) -> dict:
        """Cumulative counters plus current state. Diff two calls for a window."""
        return {
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "read_requests": self.read_requests,
            "write_requests": self.write_requests,
            "backing_reads": self.backing_reads,
            "wt_writes": self.wt_writes,
            "dirty_evictions": self.dirty_evictions,
            "flush_writes": self.flush_writes,
            "backing_writes": self.wt_writes + self.dirty_evictions + self.flush_writes,
            "switches": self.switches,
            "dirty_blocks": len(self._dirty),
            "n_objects": len(self._recency),
            "replacement": self._replacement,
            "write_policy": self._write_policy,
        }
