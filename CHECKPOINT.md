# Download and check a PixelUMM checkpoint

The default checkpoint is `S8-F22-R05` in
[nvidia/PixelUMM](https://huggingface.co/nvidia/PixelUMM). Keep model files
outside the source checkout. The checkpoint contains all learned model
weights, including the language-model weights. The Qwen3-8B config and
tokenizer are a separate, small download.

## Download the default checkpoint

Install the environment in [ENVIRONMENT.md](ENVIRONMENT.md), then run:

```bash
export PIXELUMM_MODEL_ROOT=/absolute/path/to/pixelumm-models
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/PixelUMM",
    revision="81d810cf5ba9cd7796079cd40ae32c677e34cd44",
    local_dir=os.environ["PIXELUMM_MODEL_ROOT"],
    allow_patterns=[
        "S8-F22-R05/__SAVE_COMPLETE",
        "S8-F22-R05/model/*",
    ],
)
PY
export PIXELUMM_CKPT="$PIXELUMM_MODEL_ROOT/S8-F22-R05"
```

The directory passed to `--checkpoint` contains `__SAVE_COMPLETE` and
`model/`. Preserve the hidden `model/.metadata` file and every referenced
`.distcp` shard; do not pass the `model/` subdirectory as the checkpoint path.
The loader also supports a complete `model.safetensors` checkpoint with
`__SAVE_COMPLETE`.

## Download the Qwen3-8B config and tokenizer

Only these five files are needed; do not download Qwen base-model weight
shards. Use the same local directory for every PixelUMM command:

```bash
export PIXELUMM_QWEN_DIR=/absolute/path/to/qwen3-8b-config-tokenizer
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    revision="b968826d9c46dd6066d109eabc6255188de91218",
    local_dir=os.environ["PIXELUMM_QWEN_DIR"],
    allow_patterns=[
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt",
    ],
)
PY
```

The expected SHA-256 hashes are:

| File | SHA-256 |
| --- | --- |
| `config.json` | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` |
| `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| `vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |

The tokenizer supplies the required `<|im_start|>`, `<|im_end|>`,
`<|vision_start|>`, and `<|vision_end|>` tokens. Do not add new token IDs.

## Check compatibility

```bash
CUDA_VISIBLE_DEVICES="" python check_checkpoint.py \
  --checkpoint "$PIXELUMM_CKPT" --llm-path "$PIXELUMM_QWEN_DIR"
```

This CPU-side check verifies checkpoint files and tensor names/shapes. It does
not prove that the model fits on a particular GPU; run an inference example
from [README.md](README.md) as the final check on your machine.

## F18-R01 video inference

`S8-F18-R01` is a separate checkpoint with its own model profile. Download it
from the same model repository, preserving `__SAVE_COMPLETE`, `model/.metadata`,
and all DCP shards:

```bash
export PIXELUMM_MODEL_ROOT=/absolute/path/to/pixelumm-models
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/PixelUMM",
    revision="81d810cf5ba9cd7796079cd40ae32c677e34cd44",
    local_dir=os.environ["PIXELUMM_MODEL_ROOT"],
    allow_patterns=[
        "S8-F18-R01/__SAVE_COMPLETE",
        "S8-F18-R01/model/*",
    ],
)
PY
export PIXELUMM_F18_CKPT="$PIXELUMM_MODEL_ROOT/S8-F18-R01"
```

Use the F18 profile for both the CPU checkpoint check and inference. The
following T2V example uses the landscape 480-tier size of 832×464 pixels,
96 frames, and 24 FPS:

```bash
CUDA_VISIBLE_DEVICES="" python check_checkpoint.py \
  --checkpoint "$PIXELUMM_F18_CKPT" \
  --config experiments/s8_f18_r01/release.yaml \
  --llm-path "$PIXELUMM_QWEN_DIR"

CUDA_VISIBLE_DEVICES=0 python inference.py \
  --checkpoint "$PIXELUMM_F18_CKPT" \
  --config experiments/s8_f18_r01/release.yaml \
  --llm-path "$PIXELUMM_QWEN_DIR" \
  --task t2v \
  --prompt "A golden retriever runs across a grassy field in bright daylight. The camera tracks smoothly beside the dog." \
  --height 464 --width 832 --frames 96 --fps 24 \
  --sampler unipc --steps 35 --shift 10 --cfg 6 --seed 4396 \
  --negative-prompt-file experiments/s8_f22_r07/t2v_negative_prompt.txt \
  --output /absolute/path/to/pixelumm-results/f18-t2v.mp4
```

T2V guardrails run by default. Install their separate environment using
[GUARDRAILS.md](GUARDRAILS.md), or pass `--no-guardrails` to explicitly skip
the checks.

Choose a GPU allocated to you and with sufficient memory for this resolution.
The same profile also supports text-to-image and image understanding. It does
not support `--task video-vlm`; that task needs the video-understanding
embedder present in the default F22 profile. `inference_batch.py` accepts the
same F18 config for YAML prompt batches.

For a trusted, complete DCP export that has no `__SAVE_COMPLETE` marker,
verify the publisher's file hashes first, then create an integrity receipt:

```bash
python prepare_checkpoint_export.py --checkpoint /absolute/path/to/checkpoint
```

The resulting `checkpoint_export.json` records file hashes for subsequent
loader checks; it does not modify model weights. Only load checkpoints from
trusted sources because DCP metadata uses Python serialization.
