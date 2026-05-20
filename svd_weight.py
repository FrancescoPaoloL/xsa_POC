# svd_weight.py
#
# Computes SVD-based metrics on attention weight matrices per layer.
#
# Matrices analyzed:
#   W_V          : value projection      (n_heads, d_model, d_head)
#   W_Q          : query projection      (n_heads, d_model, d_head)
#   W_K          : key projection        (n_heads, d_model, d_head)
#   W_QK         : QK circuit = W_Q @ W_K^T per head -> (d_model, d_model)
#                  determines which attention patterns the head can express
#
# Two metrics per matrix per layer, averaged across heads:
#
#   effective_rank : singular values needed to cover >= 90% of variance
#   stable_rank    : ||W||_F^2 / ||W||_2^2
#
# No forward pass required.

import torch


def compute_svd_metrics(model) -> dict[str, list[float]]:
    n_layers = model.cfg.n_layers

    results = {
        "eff_rank_V":  [0.0] * n_layers,
        "stab_rank_V": [0.0] * n_layers,
        "eff_rank_Q":  [0.0] * n_layers,
        "stab_rank_Q": [0.0] * n_layers,
        "eff_rank_K":  [0.0] * n_layers,
        "stab_rank_K": [0.0] * n_layers,
        "eff_rank_QK": [0.0] * n_layers,
        "stab_rank_QK":[0.0] * n_layers,
    }

    for layer in range(n_layers):
        WV = model.W_V[layer]  # (n_heads, d_model, d_head)
        WQ = model.W_Q[layer]
        WK = model.W_K[layer]

        results["eff_rank_V"][layer],  results["stab_rank_V"][layer]  = _metrics(WV)
        results["eff_rank_Q"][layer],  results["stab_rank_Q"][layer]  = _metrics(WQ)
        results["eff_rank_K"][layer],  results["stab_rank_K"][layer]  = _metrics(WK)
        results["eff_rank_QK"][layer], results["stab_rank_QK"][layer] = _metrics_qk(WQ, WK)

    return results


def _metrics(W: torch.Tensor) -> tuple[float, float]:
    # W: (n_heads, d_model, d_head) — SVD per head, then average.
    er_list, sr_list = [], []
    for h in range(W.shape[0]):
        s = torch.linalg.svdvals(W[h])
        er_list.append(_effective_rank(s))
        sr_list.append(_stable_rank(s))
    return sum(er_list) / len(er_list), sum(sr_list) / len(sr_list)


def _metrics_qk(WQ: torch.Tensor, WK: torch.Tensor) -> tuple[float, float]:
    # QK circuit: W_Q[h] @ W_K[h]^T -> (d_model, d_model) per head.
    er_list, sr_list = [], []
    for h in range(WQ.shape[0]):
        QK = WQ[h] @ WK[h].T  # (d_model, d_model)
        s = torch.linalg.svdvals(QK)
        er_list.append(_effective_rank(s))
        sr_list.append(_stable_rank(s))
    return sum(er_list) / len(er_list), sum(sr_list) / len(sr_list)


def _effective_rank(s: torch.Tensor, threshold: float = 0.90) -> float:
    variance = s ** 2
    total = variance.sum()
    cumulative = variance.cumsum(dim=0) / total
    return float((cumulative < threshold).sum().item()) + 1


def _stable_rank(s: torch.Tensor) -> float:
    variance = s ** 2
    return (variance.sum() / variance.max()).item()

