#!/usr/bin/env python3
"""Flush the unified buffer cache by streaming a large file through it.

`sudo purge` would be the direct route; this needs no privileges. Reading more
bytes than the machine has RAM forces every previously-cached page out.
"""
import os, sys, time

BLOCK = 8 << 20
path = sys.argv[1]
gib = float(sys.argv[2]) if len(sys.argv) > 2 else 72.0
limit = int(gib * (1 << 30))

fd = os.open(path, os.O_RDONLY)  # cached reads on purpose -- we want to evict
total = 0
t0 = time.monotonic()
while total < limit:
    chunk = os.read(fd, min(BLOCK, limit - total))
    if not chunk:
        os.lseek(fd, 0, os.SEEK_SET)
        continue
    total += len(chunk)
os.close(fd)
print(f"evicted with {total/1e9:.0f} GB in {time.monotonic()-t0:.0f}s")
