#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import glob, sys, os

result_dir = sys.argv[1]  # e.g., load_thru_20260304_120000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for f in sorted(glob.glob(f"{result_dir}/results_seqprob*")):
    seq_prob = f.split("seqprob")[-1]
    data = np.loadtxt(f)
    req_prob, avg_lat, throughput = data[:, 0], data[:, 1], data[:, 2]
    
    ax1.plot(throughput, avg_lat, 'o-', label=f"seq_prob={seq_prob}")
    ax2.plot(req_prob, throughput, 'o-', label=f"seq_prob={seq_prob}")

ax1.set_xlabel("Throughput (req/core/cycle)")
ax1.set_ylabel("Average latency (cycles)")
ax1.set_title("Load-Latency Curve")
ax1.legend()

ax2.set_xlabel("Offered load (req probability)")
ax2.set_ylabel("Throughput (req/core/cycle)")
ax2.set_title("Load-Throughput Curve")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{result_dir}/load_throughput.pdf")
plt.show()