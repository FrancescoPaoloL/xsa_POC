# Reproduces the 3-panel layout of Figure 1 in the paper, one line per run.
# Panel 4 (new): W_V effective rank per layer overlaid with attention bias (yv).
# ref: arXiv:2603.09078, Figure 1

import matplotlib.pyplot as plt

PANELS = [
    ("vv",  r"$\langle v_i, v_j \rangle$ (value similarity)"),
    ("aii", r"$a_{i,i}$ (self-attention weight)"),
    ("yv",  r"$\langle y_i, v_i \rangle$ (attention similarity bias)"),
]


def save_plot(runs: dict[str, dict[str, list[float]]], path: str) -> None:
    """runs: {label -> {"vv", "aii", "yv", "effective_rank", "stable_rank"}}."""

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

    for ax, (key, title) in zip(axes[:3], PANELS):
        for label, panels in runs.items():
            ys = panels[key]
            ax.plot(range(len(ys)), ys, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")

    # effective rank vs bias
    ax4 = axes[3]
    ax4_right = ax4.twinx()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, panels) in enumerate(runs.items()):
        er = panels["effective_rank"]
        yv = panels["yv"]
        layers = range(len(er))
        color = colors[i % len(colors)]

        ax4.plot(layers, er, marker="s", linewidth=1.5,
                 color=color, label=label)
        ax4_right.plot(layers, yv, marker="o", linewidth=1.0,
                       color=color, linestyle="--", alpha=0.5)

    ax4.set_xlabel("Layer")
    ax4.set_ylabel("W_V effective rank (solid)", fontsize=8)
    ax4_right.set_ylabel("bias ⟨y,v⟩ (dashed)", fontsize=8)
    ax4.set_title(r"$W_V$ effective rank vs bias")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(fontsize=8, loc="upper left")

    fig.suptitle("Attention similarity bias + W_V rank (ref: arXiv:2603.09078, Fig. 1)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"\nPlot saved: {path}")

