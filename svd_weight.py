# svd_weight.py
#
# Computes SVD-based metrics on attention weight matrices (W_V per layer).
#
# Two metrics per layer, averaged across heads:
#
#   effective_rank : number of singular values that capture >= 90% of total variance
#                   low value = matrix is close to low-rank
#
#   stable_rank    : ||W||_F^2 / ||W||_2^2  (= sum(s^2) / max(s)^2)
#                   smooth approximation of effective_rank, less sensitive to threshold
#
# W_V shape in TransformerLens: (n_heads, d_model, d_head)
# SVD is applied per head on the (d_model, d_head) slice.
#
# No forward pass required — weights are read directly from the model.

import torch


def compute_svd_metrics(model) -> dict[str, list[float]]:
    n_layers = model.cfg.n_layers
    eff_rank = [0.0] * n_layers
    stab_rank = [0.0] * n_layers

    for layer in range(n_layers):
        W = model.W_V[layer]  # (n_heads, d_model, d_head)
        er, sr = _svd_metrics_per_layer(W)
        eff_rank[layer] = er
        stab_rank[layer] = sr

    return {"effective_rank": eff_rank, "stable_rank": stab_rank}


def _svd_metrics_per_layer(W: torch.Tensor) -> tuple[float, float]:
    # W: (n_heads, d_model, d_head)
    # Returns averages across heads.
    er_list = []
    sr_list = []

    for head in range(W.shape[0]):
        s = torch.linalg.svdvals(W[head])  # singular values only, descending
        er_list.append(_effective_rank(s))
        sr_list.append(_stable_rank(s))

    return (
        sum(er_list) / len(er_list),
        sum(sr_list) / len(sr_list),
    )


def _effective_rank(s: torch.Tensor, threshold: float = 0.90) -> float:
    # How many singular values are needed to cover `threshold` of total variance.
    variance = s ** 2
    total = variance.sum()
    cumulative = variance.cumsum(dim=0) / total

    # first index where cumulative >= threshold, +1 for count
    rank = int((cumulative < threshold).sum().item()) + 1
    return float(rank)


def _stable_rank(s: torch.Tensor) -> float:
    # ||W||_F^2 / ||W||_2^2 = sum(s^2) / max(s)^2
    variance = s ** 2
    return (variance.sum() / variance.max()).item()
