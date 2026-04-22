# Loads real text from wikitext-2, splits it into fixed-length token chunks,
# and returns a random sample of them.
#
# prepend_bos=False: tested with and without
# no difference at SEQ_LEN=128 (the paper uses 2048, we use 128 to keep it fast on CPU)
# see config.py file

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer


def load_tokens(
    model: HookedTransformer,
    n_seqs: int,
    seq_len: int,
    seed: int,
) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n".join(t for t in dataset["text"] if t.strip())

    tokens = model.to_tokens(full_text, prepend_bos=False).squeeze(0)

    n_chunks = tokens.shape[0] // seq_len
    chunks = tokens[: n_chunks * seq_len].reshape(n_chunks, seq_len)

    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_chunks, generator=rng)[:n_seqs]
    return chunks[idx]

