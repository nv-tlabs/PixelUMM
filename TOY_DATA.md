# Toy training data

The four-step training example uses the separate
[pixelumm-toydata](https://huggingface.co/datasets/CongWei1230/pixelumm-toydata)
package. It contains local media and JSONL manifests for image generation,
video generation, image understanding, and video understanding. It is a
functional example, not a benchmark dataset.

## Download and verify

Download the ZIP outside the source checkout:

```bash
export PIXELUMM_TOY_DOWNLOAD=/absolute/path/to/toy-download
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="CongWei1230/pixelumm-toydata",
    repo_type="dataset",
    local_dir=os.environ["PIXELUMM_TOY_DOWNLOAD"],
    allow_patterns=["pixelumm-toy-v1.zip", "pixelumm-toy-v1.zip.sha256"],
)
PY

cd "$PIXELUMM_TOY_DOWNLOAD"
sha256sum -c pixelumm-toy-v1.zip.sha256
unzip -n pixelumm-toy-v1.zip -d /absolute/path/to/data
export PIXELUMM_TOY_ROOT=/absolute/path/to/data/pixelumm-toy-v1
```

The expected ZIP SHA-256 is
`6d421e1cb5c77c819db90c96c672a58c52c930ec5502c81e2927fffcf765bb71`.
Return to the PixelUMM source checkout, then verify the package inventory and
decode every media example on CPU:

```bash
cd /absolute/path/to/PixelUMM
CUDA_VISIBLE_DEVICES="" python verify_toy_data.py \
  --toy-root "$PIXELUMM_TOY_ROOT" \
  --llm-path "$PIXELUMM_QWEN_DIR" --decode
```

See [CHECKPOINT.md](CHECKPOINT.md) for `PIXELUMM_QWEN_DIR`, then follow
[TRAIN.md](TRAIN.md) for the four-step training command. The ready-to-use
package needs only the core runtime, not `requirements-data.txt`.

## Package contents

```text
pixelumm-toy-v1/
├── MANIFEST.json
├── SOURCE_PROVENANCE.json
├── t2i.jsonl
├── t2v.jsonl
├── image_vlm.jsonl
├── video_vlm.jsonl
└── media/
```

The package includes one image-generation example and ten examples for each
of the other three tasks. `MANIFEST.json` records file sizes and SHA-256
hashes. `SOURCE_PROVENANCE.json` identifies the source datasets and selected
records.
