#!/usr/bin/env python3
"""Sequential read throughput with the macOS buffer cache disabled (F_NOCACHE).

Bypassing the cache is what makes this meaningful: a model file read a second
time otherwise comes back at memory speed and tells you nothing about the disk.
"""
import fcntl, os, statistics, sys, time

F_NOCACHE = 48
BLOCK = 8 << 20  # 8 MiB, well above either disk's optimal transfer size


def read_throughput(path, limit_bytes, block=BLOCK):
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
        total = 0
        t0 = time.monotonic()
        while total < limit_bytes:
            chunk = os.read(fd, min(block, limit_bytes - total))
            if not chunk:
                break
            total += len(chunk)
        dt = time.monotonic() - t0
    finally:
        os.close(fd)
    return total, dt, total / dt / 1e9  # GB/s (decimal)


if __name__ == "__main__":
    path = sys.argv[1]
    gib = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    limit = int(gib * (1 << 30))
    rates = []
    for i in range(trials):
        total, dt, rate = read_throughput(path, limit)
        rates.append(rate)
        print(f"  trial {i+1}: {total/1e9:.1f} GB in {dt:.1f}s = {rate:.2f} GB/s", flush=True)
    print(f"median: {statistics.median(rates):.2f} GB/s")
