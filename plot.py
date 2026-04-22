# Reproduces the 3-panel layout of Figure 1 in the paper, one line per run.
# ref: arXiv:2603.09078, Figure 1

import matplotlib.pyplot as plt


PANELS = [
    ("vv",  r"$\langle v_i, v_j \rangle$  (value similarity)"),
    ("aii", r"$a_{i,i}$  (self-attention weight)"),
    ("yv",  r"$\langle y_i, v_i \rangle$  (attention similarity bias)"),
]


def save_plot(runs: dict[str, dict[str, list[float]]], path: str) -> None:
    """runs: {label -> {"vv": [...], "aii": [...], "yv": [...]}}."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for ax, (key, title) in zip(axes, PANELS):
        for label, panels in runs.items():
            ys = panels[key]
            ax.plot(range(len(ys)), ys, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Attention similarity bias multi-model  (ref: arXiv:2603.09078, Fig. 1)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"\nPlot saved: {path}")

