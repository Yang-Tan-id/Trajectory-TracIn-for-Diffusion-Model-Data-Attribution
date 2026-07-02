from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / "lds_results_without_max_score.csv")
lds = float(spearmanr(data["pred_sum_tau"], data["true_f"]).statistic)

fig, ax = plt.subplots(figsize=(12.27, 8.99), dpi=100)
ax.scatter(
    data["pred_sum_tau"],
    data["true_f"],
    s=120,
    color="#1f77b4",
    edgecolors="#1f77b4",
    alpha=0.8,
)
ax.set_title(f"LDS={lds:.4f} ({100 * lds:.2f}%)", fontsize=22, pad=10)
ax.set_xlabel("Predicted sum of attribution scores", fontsize=18)
ax.set_ylabel("True counterfactual f", fontsize=18)
ax.tick_params(axis="both", labelsize=17, length=7, width=1.3)
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(1.4)

fig.tight_layout()
fig.savefig(ROOT / "lds_scatter_without_max_score.png", dpi=100)
plt.close(fig)

print(f"LDS={lds:.12f}")
