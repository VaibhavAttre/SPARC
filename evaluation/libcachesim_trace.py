"""Convert a SPARC trace CSV into a CSV that libCacheSim can read.

SPARC trace columns:   request_id,timestamp,op,key,size,phase
libCacheSim columns:   timestamp,obj_id,obj_size,op     (1-based fields 1..4)

- `timestamp` is scaled to an integer (libCacheSim clocks are integers).
- `op` R/W becomes get/set.
- `phase` and `request_id` are dropped. Row order is preserved, so row order
  is the request order.

Usage:
    python evaluation/libcachesim_trace.py data/trace.csv data/trace.lcs.csv
    python evaluation/libcachesim_trace.py in.csv out.csv --time-scale 1000000

Reading the result (inside WSL, where libcachesim is installed):
    from libcachesim import TraceReader, TraceType, ReaderInitParam
    p = ReaderInitParam(has_header=True, has_header_set=True, delimiter=",",
                        obj_id_is_num=True, obj_id_is_num_set=True)
    p.time_field, p.obj_id_field, p.obj_size_field, p.op_field = 1, 2, 3, 4
    reader = TraceReader("data/trace.lcs.csv", TraceType.CSV_TRACE, p)
    LRU(cache_size=3).process_trace(reader)   # -> (miss_ratio_req, miss_ratio_byte)

Note: libCacheSim treats every request as a get, so it only validates
replacement (hit/miss) behavior, not write policy or dirty blocks.
"""

import argparse
import csv
import sys

OP_MAP = {"R": "get", "W": "set"}
REQUIRED = ("timestamp", "op", "key", "size")


def convert(src: str, dst: str, time_scale: int = 1_000_000) -> int:
    """Convert `src` to `dst`. Returns the number of requests written."""
    n = 0
    with open(src, newline="") as fin, open(dst, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{src}: missing required columns {missing}")
        writer = csv.writer(fout)
        writer.writerow(["timestamp", "obj_id", "obj_size", "op"])
        last_ts = None
        for line_no, row in enumerate(reader, start=2):
            op = row["op"].strip().upper()
            if op not in OP_MAP:
                raise ValueError(f"{src}:{line_no}: op must be R or W, got {row['op']!r}")
            ts = round(float(row["timestamp"]) * time_scale)
            if last_ts is not None and ts < last_ts:
                raise ValueError(f"{src}:{line_no}: timestamp decreases")
            last_ts = ts
            writer.writerow([ts, row["key"].strip(), int(row["size"]), OP_MAP[op]])
            n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", help="SPARC trace CSV")
    p.add_argument("dst", help="output libCacheSim CSV")
    p.add_argument("--time-scale", type=int, default=1_000_000,
                   help="multiplier turning timestamps into integers (default: 1e6)")
    args = p.parse_args(argv)
    n = convert(args.src, args.dst, args.time_scale)
    print(f"wrote {n} requests to {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
