# Runs MODELS, prints per-run tables, saves the combined 3-panel plot.
#
# Dependencies:
#   pip install transformer_lens matplotlib datasets
#
# ref: arXiv:2603.09078 (Exclusive Self Attention, Apple 2026)

import gc
import time

import torch
from transformer_lens import HookedTransformer

import config
from data import load_tokens
from measure import compute_panels
from report import print_run
from plot import save_plot


def main() -> None:
    runs: dict[str, dict[str, list[float]]] = {}

    for model_name in config.MODELS:
        print(f"\nLoading {model_name}...")
        model = HookedTransformer.from_pretrained(model_name)
        model.eval()

        print(f"  tokenising wikitext-2  n={config.N_SEQS}  len={config.SEQ_LEN}")
        tokens = load_tokens(model, config.N_SEQS, config.SEQ_LEN, config.SEED)

        t0 = time.perf_counter()
        panels = compute_panels(model, tokens)
        print(f"  done in {time.perf_counter() - t0:.2f}s  ({model.cfg.n_layers} layers)")

        runs[model_name] = panels
        print_run(model_name, panels)

        # free before loading the next (gpt2-large is ~3 GB in fp32)
        del model, tokens
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    save_plot(runs, config.PLOT_OUT)


if __name__ == "__main__":
    main()

