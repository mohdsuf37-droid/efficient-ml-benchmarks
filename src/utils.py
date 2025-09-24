import time
import os
from contextlib import contextmanager
import psutil
import numpy as np

@contextmanager
def timer():
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0

def file_size_mb(path):
    return os.path.getsize(path) / (1024*1024)

def measure_latency(func, warmup=10, runs=100):
    # warm-up
    for _ in range(warmup):
        func()
    times = []
    for _ in range(runs):
        with timer() as t:
            func()
        times.append(t())
    arr = np.array(times)
    return {
        "p50_ms": float(np.percentile(arr, 50)*1000),
        "p90_ms": float(np.percentile(arr, 90)*1000),
        "mean_ms": float(arr.mean()*1000)
    }

def memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024*1024)
