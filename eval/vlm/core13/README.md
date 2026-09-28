# PixelUMM Regular21

This directory contains the image-understanding evaluation suite in
`contracts/regular21.yaml`: 21 lmms-eval tasks, one task per resumable
singleton worker, greedy decoding, and the `standard_bare` assistant prefix.

The task launcher may select a subset with `--tasks` so the 21 tasks can run
independently. Use `--allow-existing-output` only when resuming the same run.

The runtime uses the checked-in model profile and a native smart-resize
minimum of 3,136 pixels with a 10,000-token multi-image budget.

Install the optional `requirements-eval.txt` layer, then run `run.py` with an
explicit `--checkpoint` and `--lmms-root`. The latter is the pinned lmms-eval
v0.7.1 package installed by `requirements-eval.txt`.
Benchmark data is read from `PIXELUMM_REGULAR21_HF_HOME`, falling back to
`PIXELUMM_VLM_HF_HOME` and then the standard Hugging Face cache.
