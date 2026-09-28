# Environment setup

PixelUMM requires Linux x86-64, Python 3.12, an NVIDIA GPU and driver
compatible with CUDA 13, the CUDA 13.0 development toolkit (`nvcc`), a C++20
compiler, and FFmpeg shared libraries for video decoding. Install the CUDA
toolkit using the [NVIDIA CUDA Linux guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
or a CUDA development image. A CUDA runtime-only image cannot build the pinned
FlashAttention source.

On Ubuntu, install FFmpeg before creating the Python environment:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ldconfig -p | grep libavutil
```

If you cannot use `sudo`, provide the CUDA toolkit and FFmpeg libraries in
your environment. Point `CUDA_HOME` to the CUDA 13.0 toolkit before building
FlashAttention. Use local scratch space for `TMPDIR` during the CUDA build.

## Install the core runtime

From the PixelUMM source directory, create the environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch-cu130.txt
python -m pip install -r requirements.txt
python -m pip install -r requirements-flash-build.txt
nvcc --version
g++ --version
FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=4 NVCC_THREADS=1 \
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  python -m pip install --no-build-isolation --no-deps -r requirements-flash-attn.txt
python -m pip check
PYTHONPATH="$PWD" python -c \
  'import torch, torchcodec, transformers; from flash_attn import flash_attn_varlen_func; import modeling.pixelumm; print("PixelUMM runtime ready")'
```

Set `FLASH_ATTN_CUDA_ARCHS` to the target GPU compute capability before the
build: `80` for A100 or RTX A6000, `90` for H100/H200, or `100` for B200.
`--no-deps` keeps the compiled FlashAttention installation from replacing the
pinned PyTorch wheel; its Python dependencies are in `requirements.txt`.
Inference uses FlashAttention varlen, while training also uses PyTorch
FlexAttention.

Check the attention kernels on the GPU you intend to use:

```bash
PYTHONPATH="$PWD" python - <<'PY'
import torch
from torch.nn.attention.flex_attention import flex_attention
from flash_attn import flash_attn_varlen_func

assert torch.cuda.is_available(), "CUDA GPU not visible"
q = torch.randn(8, 4, 64, device="cuda", dtype=torch.bfloat16)
cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
out = flash_attn_varlen_func(q, q, q, cu, cu, 8, 8)
assert out.shape == q.shape and torch.isfinite(out).all()
packed = q.permute(1, 0, 2).unsqueeze(0).contiguous()
flex_out = flex_attention(packed, packed, packed)
assert flex_out.shape == packed.shape and torch.isfinite(flex_out).all()
print("Attention kernels ready:", torch.cuda.get_device_name())
PY
```

Before video inference or toy training, also test a real MP4 from the
[toy data package](TOY_DATA.md). `imageio-ffmpeg` provides an encoding binary
but not the shared FFmpeg libraries used by TorchCodec for decoding:

```bash
python - <<'PY'
from torchcodec.decoders import VideoDecoder
video = VideoDecoder("/absolute/path/to/data/pixelumm-toy-v1/media/t2v/r05_sana_008.mp4")
assert len(video) > 0 and video[0].ndim == 3
print("Video decoding ready:", len(video), "frames")
PY
```

## Optional packages

Install `requirements-eval.txt` only for the optional benchmark tools in
[EVAL.md](EVAL.md). `requirements-data.txt` is used only when rebuilding the
toy package from its source datasets; the ready-to-use toy package does not
need it.
