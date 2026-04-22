# Measures three quantities per attention layer in a single forward pass:
#
#   1) <v_i, v_j>  : how similar value vectors are to each other
#   2) a_{i,i}     : how much each token attends to itself
#   3) <y_i, v_i>  : how similar the attention output is to the token's own value
#
# Each hook computes its average on the spot and stores just a float,
# not the full tensor of shape (B, T, H, D):
#   B = batch size, T = sequence length, H = heads, D = head dimension
# This way only one layer lives in RAM at a time instead of all layers at once.
#
# ref: arXiv:2603.09078, Figure 1
# ref: https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def compute_panels(model: HookedTransformer, tokens: torch.Tensor) -> dict[str, list[float]]:
    # Run the model once and collect all three measurements for every layer.
    n_layers = model.cfg.n_layers
    vv  = [0.0] * n_layers
    aii = [0.0] * n_layers
    yv  = [0.0] * n_layers
    v_buf: dict[int, torch.Tensor] = {}

    hooks = []
    for layer in range(n_layers):
        hooks.append((f"blocks.{layer}.attn.hook_v",       _v_hook(layer, v_buf, vv)))
        hooks.append((f"blocks.{layer}.attn.hook_pattern", _p_hook(layer, aii)))
        hooks.append((f"blocks.{layer}.attn.hook_z",       _z_hook(layer, v_buf, yv)))

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks)

    return {"vv": vv, "aii": aii, "yv": yv}


def _v_hook(l, v_buf, vv):
    # Called when value vectors are ready. Save them for later and measure panel 1.
    # v shape: (B, T, H, D).
    def fn(v, hook):
        v_buf[l] = v
        vv[l] = _avg_pairwise_cos(v)
    return fn


def _p_hook(l, aii):
    # Called when the attention pattern is ready. Read the diagonal to get a_{i,i}.
    # p shape: (B, H, T, T).
    def fn(p, hook):
        aii[l] = p.diagonal(dim1=-2, dim2=-1).mean().item()
    return fn


def _z_hook(l, v_buf, yv):
    # Called when the attention output is ready. Compare it with the saved value vector.
    # z shape: (B, T, H, D).
    def fn(z, hook):
        v = v_buf.pop(l)
        yv[l] = F.cosine_similarity(z, v, dim=-1).mean().item()
    return fn


def _avg_pairwise_cos(v: torch.Tensor) -> float:
    # Measure how similar value vectors are to each other within the same sequence.
    # Only pairs i != j are counted (a vector is always identical to itself).
    v = v.transpose(1, 2)                  # (B, T, H, D) -> (B, H, T, D)
    vn = F.normalize(v, dim=-1)
    sim = vn @ vn.transpose(-2, -1)        # (B, H, T, T)
    T = sim.shape[-1]
    mask = 1.0 - torch.eye(T, device=sim.device)
    return (sim * mask).sum(dim=(-2, -1)).div(T * (T - 1)).mean().item()

