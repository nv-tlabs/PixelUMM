# PixelUMM Video4

This directory contains the video-understanding evaluation suite:
`contract_r07_video4.yaml` with MVBench, Video-MME without subtitles,
LongVideoBench validation-video-only, and LVBench.

The input contract is deterministic `short_image`: strict 1 FPS with at most
96 unique frames when the selected duration is at most 96 seconds; longer
clips use full-clip uniform sampling with at most 96 unique frames. Source FPS
outside [1, 240] is rejected. Frames retain native aspect ratio and use the
smart resize budget of at most 200,704 pixels per frame.

Each task can run as eight resumable singleton shards. The runner supports
task selection, sharding, and `--resume`.

The runtime uses the video-understanding token path and the checked-in model
and video preprocessing profiles.

Install the optional `requirements-eval.txt` layer, then run `run.py` with an
explicit `--checkpoint` and `--lmms-root`. The latter is the pinned lmms-eval
v0.7.1 package installed by `requirements-eval.txt`.
Benchmark data is read from `PIXELUMM_VIDEO4_HF_HOME`, falling back to
`PIXELUMM_VLM_HF_HOME` and then the standard Hugging Face cache.
