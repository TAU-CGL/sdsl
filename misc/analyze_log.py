import re
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

LOG_FILE = "log.txt"

pattern = re.compile(r"Pixel at \((\d+), (\d+)\): \d+\s+negate:\w+ occ: ([\d.]+)")

rows, cols, occs = [], [], []
with open(LOG_FILE) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            rows.append(int(m.group(1)))
            cols.append(int(m.group(2)))
            occs.append(float(m.group(3)))

counts = Counter(occs)
print({k: counts[k] for k in sorted(counts)})

unique_vals = sorted(counts)
cmap = cm.get_cmap("tab10", len(unique_vals))
color_map = {v: cmap(i) for i, v in enumerate(unique_vals)}

fig, ax = plt.subplots(figsize=(10, 10))
for uv in unique_vals:
    xs = [cols[i] for i in range(len(occs)) if occs[i] == uv]
    ys = [rows[i] for i in range(len(occs)) if occs[i] == uv]
    ax.scatter(xs, ys, s=1, color=color_map[uv], label=f"occ={uv:.4f} (n={counts[uv]})")

ax.set_xlabel("col")
ax.set_ylabel("row")
ax.invert_yaxis()
ax.legend(markerscale=8, loc="best")
ax.set_title("Pixel occupancy map")
plt.tight_layout()
plt.savefig("occ_map.png", dpi=150)
plt.show()
