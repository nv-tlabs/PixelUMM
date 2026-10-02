<h1 align="center">PixelUMM: Encoder-Free Unified Image and Video Understanding and Generation</h1>

<p align="center"><strong>
  <a href="https://congwei1230.github.io/">Cong Wei<sup>1,2</sup></a> &ensp;
  <a href="https://xuanchiren.com/">Xuanchi Ren<sup>1</sup></a> &ensp;
  <a href="https://github.com/nv-tlabs/PixelUMM">Bryan Chu<sup>1</sup></a> &ensp;
  <a href="https://cs.uwaterloo.ca/~w2ren/">Weiming Ren<sup>2</sup></a><br>
  <a href="https://www.cs.toronto.edu/~linghuan/">Huan Ling<sup>1</sup></a> &ensp;
  <a href="https://huangjh-pub.github.io/">Jiahui Huang<sup>1</sup></a> &ensp;
  <a href="https://dvl.in.tum.de/team/lealtaixe/">Laura Leal-Taixé<sup>1</sup></a> &ensp;
  <a href="https://www.cs.toronto.edu/~fidler/">Sanja Fidler<sup>1</sup></a><br>
  <a href="https://wenhuchen.github.io/">Wenhu Chen<sup>2</sup></a> &ensp;
  <a href="https://www.cs.toronto.edu/~zianwang/">Zian Wang<sup>1</sup></a> &ensp;
  <a href="https://zhangjiewu.github.io/">Jay Zhangjie Wu<sup>1</sup></a>
</strong></p>

<p align="center"><sup>1</sup>NVIDIA &nbsp;&nbsp; <sup>2</sup>University of Waterloo</p>

<p align="center">
  <a href="https://congwei1230.github.io/pixelumm-project-page-preview/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2609.38597"><img src="https://img.shields.io/badge/arXiv-2609.38597-b31b1b?logo=arxiv" alt="arXiv:2609.38597"></a>
  &nbsp;
  <a href="https://huggingface.co/papers/2609.38597"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-orange" alt="Hugging Face Paper"></a>
  &nbsp;
  <a href="https://huggingface.co/nvidia/PixelUMM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange" alt="Hugging Face Model"></a>
</p>

<p align="center">
  <a href="https://congwei1230.github.io/pixelumm-project-page-preview/media/overview.mp4"><img src="assets/overview.gif" alt="30-second PixelUMM architecture overview"></a>
</p>

PixelUMM reads and writes raw pixels with a single decoder-only Transformer — no VAE and no vision encoder. Images become 16×16 patches and videos become 4-frame tubes; a Qwen3-8B backbone with separate understanding and generation experts shares one self-attention across text, clean pixels, and noisy pixels.

## Quick start

This repository provides inference code for image and video generation and
understanding, plus a four-step toy training example. `S8-F22-R05` is the
default checkpoint after Stage 2 training and is the checkpoint used in the
paper evaluation. `S8-F18-R01` is an intermediate checkpoint that received
10K additional fine-tuning steps at 480p and 720p short-side settings. It
generally gives slightly better text-to-video results than the default
checkpoint. Model weights and toy data are downloaded separately
from this source repository.

1. Clone the repository and install the CUDA environment using
   [ENVIRONMENT.md](ENVIRONMENT.md). For T2V, also set up the separate
   [Cosmos guardrail environment](GUARDRAILS.md); checks are on by default.
2. Download `S8-F22-R05` from
   [nvidia/PixelUMM](https://huggingface.co/nvidia/PixelUMM) and the Qwen3-8B
   config/tokenizer files using [CHECKPOINT.md](CHECKPOINT.md). The PixelUMM
   checkpoint contains the learned language-model weights; separate Qwen
   weight shards are not needed.
3. Set the local paths and check the checkpoint before running on a GPU:

```bash
export PIXELUMM_CKPT=/absolute/path/to/PixelUMM/S8-F22-R05
export PIXELUMM_QWEN_DIR=/absolute/path/to/qwen3-8b-config-tokenizer
export PIXELUMM_OUTPUT=/absolute/path/to/pixelumm-results

CUDA_VISIBLE_DEVICES="" python check_checkpoint.py \
  --checkpoint "$PIXELUMM_CKPT" --llm-path "$PIXELUMM_QWEN_DIR"
```

`check_checkpoint.py` checks checkpoint completeness and tensor compatibility
without loading the full model on a GPU. Run the examples below on a GPU
allocated to you; set `CUDA_VISIBLE_DEVICES` accordingly.

## Inference

For the default T2V safety checks, first request access to
[`nvidia/Cosmos-1.0-Guardrail`](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail),
accept its access conditions, and sign in with the approved Hugging Face
account as described in [GUARDRAILS.md](GUARDRAILS.md). A Hugging Face login
alone does not grant access to this separately gated model. Its weights are
downloaded on first use.

```bash
# Text to image
python inference.py \
  --checkpoint "$PIXELUMM_CKPT" --llm-path "$PIXELUMM_QWEN_DIR" \
  --task t2i --prompt "A red panda reading beside a window" \
  --height 256 --width 256 --seed 4396 \
  --output "$PIXELUMM_OUTPUT/t2i.png"

# Text to video: 96 frames, 24 FPS
python inference.py \
  --checkpoint "$PIXELUMM_CKPT" --llm-path "$PIXELUMM_QWEN_DIR" \
  --task t2v \
  --prompt "A golden retriever runs across a grassy field in bright daylight. The camera tracks smoothly beside the dog." \
  --height 176 --width 320 --frames 96 --fps 24 --steps 35 \
  --negative-prompt-file experiments/s8_f22_r07/t2v_negative_prompt.txt \
  --output "$PIXELUMM_OUTPUT/t2v.mp4"

# Image understanding
python inference.py \
  --checkpoint "$PIXELUMM_CKPT" --llm-path "$PIXELUMM_QWEN_DIR" \
  --task image-vlm --image /absolute/path/to/photo.jpg \
  --prompt "Describe the image." \
  --output "$PIXELUMM_OUTPUT/image-answer.txt"

# Video understanding
python inference.py \
  --checkpoint "$PIXELUMM_CKPT" --llm-path "$PIXELUMM_QWEN_DIR" \
  --task video-vlm --video /absolute/path/to/clip.mp4 \
  --prompt "Describe the video." \
  --output "$PIXELUMM_OUTPUT/video-answer.txt"
```

The default F22 checkpoint supports all four tasks. `S8-F18-R01` supports
image/video generation and image understanding, but not video understanding.
Its separate download, model profile, and 832×464 T2V inference command are in
[CHECKPOINT.md](CHECKPOINT.md#f18-r01-video-inference).

For a YAML prompt batch, use `inference_batch.py --help`. The output directory
must be new for each run. This batch entrypoint does not run guardrails; the
default checks currently apply to single-command T2V only. For benchmark
evaluation, see [EVAL.md](EVAL.md). Pass `--no-guardrails` to the single-command
T2V entrypoint only when you explicitly intend to skip its safety checks.

## Four-step toy training

Download and verify the [toy data package](TOY_DATA.md), then run the complete
four-task example in [TRAIN.md](TRAIN.md). The example starts from the F22
checkpoint and saves model weights that `inference.py` can load. It is a
functional training example, not a full training recipe or an optimizer-state
resume. Plan for seven GPUs with at least 48 GiB each and sufficient disk
space for an approximately 61 GB output checkpoint.

## Repository layout

- `modeling/`: model architecture and generation modules.
- `data/`: local multimodal dataset and preprocessing.
- `train/`, `train_toy.py`: model loading and toy training.
- `inference.py`, `inference_batch.py`: inference entrypoints.
- `experiments/`: model profiles and generation settings.
- `eval/`: optional evaluation tools.

Most source files are licensed under [Apache-2.0](LICENSE). File-specific
notices are retained where upstream code has different terms; in particular,
`modeling/pixelumm/modeling_utils.py` retains its DiT-derived CC BY-NC 4.0
notice. Model weights have separate terms in the model repository.

## Citation

If you use PixelUMM, please cite our paper:

```bibtex
@misc{wei2026pixelummencoderfreeunifiedimage,
  title={PixelUMM: Encoder-Free Unified Image and Video Understanding and Generation},
  author={Cong Wei and Xuanchi Ren and Bryan Chu and Weiming Ren and Huan Ling and Jiahui Huang and Laura Leal-Taixé and Sanja Fidler and Wenhu Chen and Zian Wang and Jay Zhangjie Wu},
  year={2026},
  eprint={2609.38597},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.38597},
}
```
