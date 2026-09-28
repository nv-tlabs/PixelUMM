# Cosmos guardrails for video generation

Single-command text-to-video inference runs Cosmos guardrails by default.
`inference.py` checks the prompt before loading PixelUMM and checks every
generated frame before writing the requested output. The checks use the Cosmos
blocklist, Qwen3Guard, and video content
classifier. Safe videos are then processed by the RetinaFace face-pixelation
filter. A rejected prompt or video produces no final output file; a missing
model or failed check is an error, not an approval.

## Access and setup

Before T2V inference, request access to
[`nvidia/Cosmos-1.0-Guardrail`](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
and accept its access conditions. Authenticate with the same Hugging Face
account after access is granted. Signing in by itself does not grant access to
the gated repository. Guardrail weights are separate from the PixelUMM
checkpoint and are downloaded on first use, along with the public Qwen3Guard
and SigLIP models. Allow time and disk space for those downloads.

The guardrail code needs a separate Python environment because its
`transformers` major version differs from PixelUMM's. First install the normal
PixelUMM CUDA environment described in [ENVIRONMENT.md](ENVIRONMENT.md). On
Ubuntu, install the system libraries needed by the video face filter, then
create a second Python 3.12 environment at the repository root:

```bash
sudo apt-get install -y libgl1 libglib2.0-0 libxcb1
python3.12 -m venv .venv-guardrails
.venv-guardrails/bin/python -m pip install -r requirements-torch-cu130.txt
.venv-guardrails/bin/python -m pip install -r requirements-guardrails.txt
.venv-guardrails/bin/python -m pip check
.venv-guardrails/bin/hf auth login
```

On a system without `sudo`, provide the corresponding OpenCV system libraries
in the runtime image. The Cosmos weights remain subject to their own model
license.

## Inference

Run a regular T2V command; `.venv-guardrails/bin/python` is found automatically:

```bash
python inference.py \
  --checkpoint "$PIXELUMM_CKPT" --llm-path "$PIXELUMM_QWEN_DIR" \
  --task t2v --prompt "A dog runs across a grassy field." \
  --height 176 --width 320 --frames 96 --fps 24 \
  --output "$PIXELUMM_OUTPUT/guarded-t2v.mp4"
```

The same default applies to the F18-R01 T2V command in
[CHECKPOINT.md](CHECKPOINT.md#f18-r01-video-inference). To place the guardrail
environment elsewhere, set `PIXELUMM_GUARDRAILS_PYTHON` to its Python path or
pass `--cosmos-guardrails-python /absolute/path/to/python`. To explicitly skip
the checks, pass `--no-guardrails`; this also avoids loading guardrail weights.
Users who disable safety checks remain responsible for complying with the
applicable model license and for reviewing their outputs.

The output path must be new. Video that passes the checks is re-encoded after face processing,
so its pixels and file hash will differ from an unguarded generation even when
the prompt, seed, and PixelUMM sampler are unchanged. The guardrail is a
content-safety check, not a visual-quality or temporal-stability repair tool.
The classifier can reject benign content and does not guarantee that every
unsafe output will be detected; review outputs before publication.
The checks currently cover only single-command T2V inference. T2I and batch
generation do not run them; VLM tasks and training are outside this output
safety path.
