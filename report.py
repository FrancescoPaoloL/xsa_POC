# Per-run table for the three panels + SVD metrics + trend summary.
# ref: arXiv:2603.09078, Figure 1

def print_run(label: str, panels: dict[str, list[float]]) -> None:
    vv  = panels["vv"]
    aii = panels["aii"]
    yv  = panels["yv"]
    er_v  = panels["eff_rank_V"]
    er_q  = panels["eff_rank_Q"]
    er_k  = panels["eff_rank_K"]
    er_qk = panels["eff_rank_QK"]
    n = len(yv)

    print(f"\n=== {label} ===")
    print(f"  {'Layer':<6} {'<v_i,v_j>':>12} {'a_{i,i}':>12} {'<y_i,v_i>':>12} "
          f"{'er_V':>8} {'er_Q':>8} {'er_K':>8} {'er_QK':>8}")
    print(f"  {'-' * 82}")

    for i in range(n):
        print(
            f"  {i:<6} {vv[i]:>12.4f} {aii[i]:>12.4f} {yv[i]:>12.4f} "
            f"{er_v[i]:>8.1f} {er_q[i]:>8.1f} {er_k[i]:>8.1f} {er_qk[i]:>8.1f}"
        )

    print("\n  trend (last - first):")
    for key, name in [
        ("vv",          "<v_i,v_j>"),
        ("aii",         "a_{i,i}"),
        ("yv",          "<y_i,v_i>"),
        ("eff_rank_V",  "er_V"),
        ("eff_rank_Q",  "er_Q"),
        ("eff_rank_K",  "er_K"),
        ("eff_rank_QK", "er_QK"),
    ]:
        xs = panels[key]
        t  = xs[-1] - xs[0]
        arrow = "↑" if t > 0.05 else ("↓" if t < -0.05 else "·")
        print(f"  {name:<12} {t:+.4f} {arrow}")

