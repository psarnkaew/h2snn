"""Run a command and print its peak RSS (children's ru_maxrss) and wall time (card baselines-h2gb job wrapper)."""
import resource
import subprocess
import sys
import time

start = time.time()
status = subprocess.call(sys.argv[1:])
peak_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(f"peak_rss_gib={peak_kib / 2**20:.2f} elapsed_s={time.time() - start:.1f}", flush=True)
sys.exit(status)
