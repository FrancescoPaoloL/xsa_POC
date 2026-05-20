# Reproduces the 3-panel layout of Figure 1 in the paper (row 1).
# Row 2: SVD effective rank for W_Q, W_K, W_QK overlaid with bias (yv).
# ref: arXiv:2603.09078, Figure 1

import matplotlib.pyplot as plt

PANELS = [
    ("vv",  r"$\langle v_i, v_j \rangle$ (value similarity)"),
    ("aii", r"$a_{i,i}$ (self-attention weight)"),
    ("yv",  r"$\langle y_i, v_i \rangle$ (attention similarity bias)"),
]

SVD_PANELS = [
    ("eff_rank_V",  r"$W_V$ eff. rank vs bias",      (0, 3)),
    ("eff_rank_Q",  r"$W_Q$ eff. rank vs bias",      (1, 0)),
    ("eff_rank_K",  r"$W_K$ eff. rank vs bias",      (1, 1)),
    ("eff_rank_QK", r"$W_Q W_K^T$ eff. rank vs bias",(1, 2)),
]


def save_plot(runs: dict[str, dict[str, list[float]]], path: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (key, title) in zip(axes[0, :3], PANELS):
        for label, panels in runs.items():
            ys = panels[key]
            ax.plot(range(len(ys)), ys, marker="o", linewidth=1.5, label=label)
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    # SVD panels: explicit position per panel
    for rank_key, title, (row, col) in SVD_PANELS:
        ax = axes[row, col]
        ax_right = ax.twinx()
        for i, (label, panels) in enumerate(runs.items()):
            er  = panels[rank_key]
            yv  = panels["yv"]
            color = colors[i % len(colors)]
            ax.plot(range(len(er)), er, marker="s", linewidth=1.5,
                    color=color, label=label)
            ax_right.plot(range(len(yv)), yv, marker="o", linewidth=1.0,
                          color=color, linestyle="--", alpha=0.4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("eff. rank (solid)", fontsize=8)
        ax_right.set_ylabel("bias (dashed)", fontsize=8)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left")

    axes[1, 3].set_visible(False)

    fig.suptitle("Attention similarity bias + W SVD rank (ref: arXiv:2603.09078, Fig. 1)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"\nPlot saved: {path}")

