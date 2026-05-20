# Per-run table for the three panels + SVD metrics + trend summary.
# ref: arXiv:2603.09078, Figure 1

def print_run(label: str, panels: dict[str, list[float]]) -> None:
    vv  = panels["vv"]
    aii = panels["aii"]
    yv  = panels["yv"]
    er  = panels["effective_rank"]
    sr  = panels["stable_rank"]
    n   = len(yv)

    print(f"\n=== {label} ===")
    print(f"  {'Layer':<6} {'<v_i,v_j>':>12} {'a_{i,i}':>12} {'<y_i,v_i>':>12} {'eff_rank':>10} {'stab_rank':>10}")
    print(f"  {'-' * 66}")

    for i in range(n):
        print(
            f"  {i:<6} {vv[i]:>12.4f} {aii[i]:>12.4f} "
            f"{yv[i]:>12.4f} {er[i]:>10.1f} {sr[i]:>10.2f}"
        )

    print("\n  trend (last - first):")
    for key, name in [
        ("vv",             "<v_i,v_j>"),
        ("aii",            "a_{i,i}"),
        ("yv",             "<y_i,v_i>"),
        ("effective_rank", "eff_rank"),
        ("stable_rank",    "stab_rank"),
    ]:
        xs = panels[key]
        t  = xs[-1] - xs[0]
        arrow = "↑" if t > 0.05 else ("↓" if t < -0.05 else "·")
        print(f"  {name:<12} {t:+.4f} {arrow}")

