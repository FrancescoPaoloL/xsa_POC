# xsa_POC

A minimal experiment that replicates Figure 1 of the paper
*Exclusive Self Attention* (Zhai, Apple 2026  arXiv:2603.09078)
on GPT-2 variants using TransformerLens.

**This is a learning project.**
Not production-ready, not optimized.

## Why

The paper identifies an *attention similarity bias* in standard Transformers:
the attention output $y_i$ ends up pointing in the same direction as the
token's own value vector $v_i$, layer after layer.
This means SA wastes capacity copying a token's own information instead of
aggregating context  which is its actual job.

This POC measures that bias on real models to see whether the paper's findings
hold at smaller scale and on a different training setup (WebText vs FineWeb).

## What it measures

Three quantities from Figure 1 of the paper, for each attention layer:

| Panel | Quantity | What it tells you |
|-------|----------|-------------------|
| Left | $\langle v_i, v_j \rangle$  avg pairwise cosine sim of value vectors | how correlated value vectors are within a sequence |
| Middle | $a_{i,i}$  avg diagonal of attention pattern | how much each token attends to itself |
| Right | $\langle y_i, v_i \rangle$  avg cosine sim of attention output and self value | the bias itself |

The attention similarity bias is defined as the cosine similarity between
the attention output and the token's own value vector:

```math
\langle y_i, v_i \rangle = \frac{y_i \cdot v_i}{\|y_i\| \|v_i\|}
```

where $y_i = \sum_{j=1}^{i} a_{i,j} v_j$ is the standard SA output.

## Setup

```bash
pip install transformer_lens matplotlib datasets
python main.py
```

No GPU required. Runs on CPU; expect ~1–3 min for all three models.

## Models

Configured in `config.py`:

```python
MODELS  = ["gpt2", "gpt2-medium", "gpt2-large"]  # 117M, 345M, 774M
N_SEQS  = 32
SEQ_LEN = 128
```

Input: 32 sequences × 128 tokens sampled from the wikitext-2 test split.

## Results

The bias is present across all three models, but the shape diverges from
the paper. The paper (1.3B, RoPE, FineWeb) reports a **monotonically
increasing** trend with depth. GPT-2 shows a **U-shape**: bias is highest
in early layers, drops toward the middle, then rises slightly at the end.

| Model | Layer 0 | Min | Last layer |
|-------|---------|-----|------------|
| gpt2 (117M, 12L) | 0.72 | ~0.35 (L5) | 0.50 |
| gpt2-medium (345M, 24L) | 0.63 | ~0.35 (L19) | 0.60 |
| gpt2-large (774M, 36L) | 0.51 | ~0.32 (L28) | 0.59 |

The minimum shifts to roughly 75–80% of depth across all three models.
$a_{i,i}$ is low (<0.05) in deep layers  GPT-2 does not self-attend heavily
in depth. The late-layer rise in $\langle y_i, v_i \rangle$ appears to come
from value vector geometry (correlated $v_j$) rather than from
self-attention directly.

Possible causes of the divergence: WebText + learned positional embeddings
(GPT-2) vs FineWeb + RoPE (paper); SEQ_LEN 128 vs 2048; training recipe.

## Files

```
config.py    models, sequence count, seed, output path
data.py      wikitext-2 loader, reproducible chunking
measure.py   single forward pass, hooks on hook_v / hook_pattern / hook_z
report.py    per-layer table with trend arrows
plot.py      3-panel figure matching paper layout
main.py      loop over models, gc between runs
```

## References

Zhai  Exclusive Self Attention, arXiv:2603.09078, Apple 2026.
arxiv.org/abs/2603.09078

Nanda et al.  TransformerLens.
github.com/TransformerLensOrg/TransformerLens

## Connect with me
<p align="left">
<a href="https://www.linkedin.com/in/francescopl/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="francescopaololezza" height="20" width="30" /></a>
<a href="https://www.kaggle.com/francescopaolol" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/kaggle.svg" alt="francescopaololezza" height="20" width="30" /></a>
</p>

