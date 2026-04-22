# Per-run table for the three panels + trend summary.
# ref: arXiv:2603.09078, Figure 1


def print_run(label: str, panels: dict[str, list[float]]) -> None:
    vv, aii, yv = panels["vv"], panels["aii"], panels["yv"]
    n = len(yv)

    print(f"\n=== {label} ===")
    print(f"  {'Layer':<6} {'<v_i,v_j>':>12} {'a_{i,i}':>12} {'<y_i,v_i>':>12}")
    print(f"  {'-' * 46}")
    for i in range(n):
        print(f"  {i:<6} {vv[i]:>12.4f} {aii[i]:>12.4f} {yv[i]:>12.4f}")

    print("\n  trend (last - first):")
    for key, name in [("vv", "<v_i,v_j>"), ("aii", "a_{i,i}"), ("yv", "<y_i,v_i>")]:
        xs = panels[key]
        t = xs[-1] - xs[0]
        arrow = "↑" if t > 0.05 else ("↓" if t < -0.05 else "·")
        print(f"    {name:<10}  {t:+.4f}  {arrow}")

