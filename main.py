# Runs MODELS, prints per-run tables, saves the combined plot.
#
# Dependencies:
# pip install transformer_lens matplotlib datasets
#
# ref: arXiv:2603.09078 (Exclusive Self Attention, Apple 2026)

import gc
import time
import torch
from transformer_lens import HookedTransformer

import config
from data import load_tokens
from measure import compute_panels
from svd_weight import compute_svd_metrics  # <-- nuovo
from report import print_run
from plot import save_plot


def main() -> None:
    runs: dict[str, dict[str, list[float]]] = {}

    for model_name in config.MODELS:
        print(f"\nLoading {model_name}...")
        model = HookedTransformer.from_pretrained(model_name)
        model.eval()

        print(f"  tokenising wikitext-2 n={config.N_SEQS} len={config.SEQ_LEN}")
        tokens = load_tokens(model, config.N_SEQS, config.SEQ_LEN, config.SEED)

        t0 = time.perf_counter()
        panels = compute_panels(model, tokens)
        print(f"  attention done in {time.perf_counter() - t0:.2f}s ({model.cfg.n_layers} layers)")

        t1 = time.perf_counter()
        svd = compute_svd_metrics(model)  # <-- nessun forward pass, solo pesi
        print(f"  svd done in {time.perf_counter() - t1:.2f}s")

        panels.update(svd)  # aggiunge effective_rank e stable_rank a panels
        print("  svd keys:", list(svd.keys()))
        print("  eff_rank_QK sample:", svd["eff_rank_QK"][:3])
        runs[model_name] = panels

        print_run(model_name, panels)

        del model, tokens
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    save_plot(runs, config.PLOT_OUT)


if __name__ == "__main__":
    main()

