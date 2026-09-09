## Review
- **Correct:** Pair-specific pooling uses drug `i` and raw protein `j`; gathers occur inside the checkpoint (`src/spikes/phase1/fragment_encoder.py:174–197,223–238`).
- **Correct:** `finally` restores aligned F2R attribution, including interrupted checkpoint recomputation (`fragment_encoder.py:180–191`). Aligned outputs remain separate from off-diagonal conditioning.
- **Correct:** Tests cover independent-pair forward/gradient equivalence, attribution, permutation/padding/chunk invariance, optimizer parity, and wrappers (`tests/test_fragment_pair_conditioning.py:42–188`).
- **Blocker:** None identified.
- **Note — Low:** Conditioned forwards still compute an unused aligned drug projection/self-attention before recomputing it per pair (`src/spikes/phase1/fragment_encoder.py:263`). Minor avoidable overhead, not a correctness issue.
- **Residual risks:** CPU-double tests do not establish CUDA/autocast behavior or realistic peak memory. No saved-tensor regression explicitly guards against moving raw-protein gathers outside checkpointing. Runtime validation remains with the parent.