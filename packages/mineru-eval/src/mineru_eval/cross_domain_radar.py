"""
Cross-Domain Radar Chart Generator (S3*)

Generates a radar/spider chart comparing 3 general-purpose small models
against the qwen3.5:9b baseline across 4 dimensions.

Data source: cross_domain_comparison.md (original _report.json lost in Remove-Item accident)
"""

import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DIMENSIONS = ["A\nConceptual", "B\nMulti-Step", "C\nSensitivity", "D\nDesign"]
N_DIMS = len(DIMENSIONS)

MODELS = [
    {"name": "Qwen3.5-9B (baseline)", "scores": [0.5688, 0.6104, 0.6620, 0.6586], "color": "#1a56db", "linestyle": "-", "linewidth": 2.5},
    {"name": "Qwen3.5-2B",            "scores": [0.3200, 0.3067, 0.3667, 0.3333], "color": "#e37a22", "linestyle": "--", "linewidth": 1.8},
    {"name": "Gemma3-4B",             "scores": [0.3067, 0.1067, 0.4000, 0.2280], "color": "#16a34a", "linestyle": "--", "linewidth": 1.8},
    {"name": "Llama3.2-3B",           "scores": [0.1600, 0.0400, 0.2500, 0.1320], "color": "#9333ea", "linestyle": "--", "linewidth": 1.8},
]

ANGLES = np.linspace(0, 2 * np.pi, N_DIMS, endpoint=False).tolist()
ANGLES += ANGLES[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(ANGLES[:-1])
ax.set_xticklabels(DIMENSIONS, fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="gray")
ax.set_rlabel_position(30)

for model in MODELS:
    values = model["scores"] + model["scores"][:1]
    ax.plot(ANGLES, values, color=model["color"], linestyle=model["linestyle"],
            linewidth=model["linewidth"], label=model["name"])
    ax.fill(ANGLES, values, color=model["color"], alpha=0.06)

ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9)
ax.set_title("Cross-Domain Generalization: 4-Dimension Radar Comparison",
             fontsize=13, fontweight="bold", pad=20)

fig.text(0.5, 0.02, "Data: ControlSci core 60 balanced subset (15 per dimension, seed=42) | Judge: deepseek-v4-flash",
         ha="center", fontsize=9, color="gray")

output_dir = Path(__file__).resolve().parents[2] / "benchmark" / "eval" / "results"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "cross_domain_radar.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Radar chart saved: {output_path}")
