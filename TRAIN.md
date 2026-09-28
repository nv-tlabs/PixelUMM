# Four-step toy training

This example starts from the default `S8-F22-R05` model, runs one optimizer
step for each of the four tasks, and saves model weights that can be loaded by
`inference.py`. It demonstrates the training interface; it is not a
convergence or quality benchmark.

Install the core runtime using [ENVIRONMENT.md](ENVIRONMENT.md), download and
check the model and tokenizer using [CHECKPOINT.md](CHECKPOINT.md), and
download and verify the separate [toy data package](TOY_DATA.md).

## Run

The example requires seven GPUs with at least 48 GiB each. Adjust
`CUDA_VISIBLE_DEVICES` to GPUs allocated to your job. Keep the checkpoint,
data, and output paths outside the source checkout.

```bash
export PIXELUMM_CKPT=/absolute/path/to/PixelUMM/S8-F22-R05
export PIXELUMM_QWEN_DIR=/absolute/path/to/qwen3-8b-config-tokenizer
export PIXELUMM_TOY_ROOT=/absolute/path/to/data/pixelumm-toy-v1
export PIXELUMM_TRAIN_OUTPUT=/absolute/path/to/pixelumm-results/toy-train
export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
torchrun --standalone --nproc_per_node=7 train_toy.py \
  --checkpoint "$PIXELUMM_CKPT" \
  --llm-path "$PIXELUMM_QWEN_DIR" \
  --toy-root "$PIXELUMM_TOY_ROOT" \
  --output "$PIXELUMM_TRAIN_OUTPUT" \
  --steps 4 --expected-num-tokens 1 --max-num-tokens 20000
```

Use a new output directory for each run. The small token budget places one
real example on each rank per step; it does not reduce the model size or video
frame count. The four steps cover T2I, T2V, image VLM, and video VLM. The
example uses FSDP full sharding, BF16 computation, FP32 master parameters,
activation checkpointing, and a fresh AdamW optimizer. Training is more
memory-intensive than inference, and other GPU configurations need separate
capacity checks.

`metrics.jsonl` records task, losses, gradient norms, and sampled parameter
updates. The completed checkpoint is written to
`$PIXELUMM_TRAIN_OUTPUT/checkpoints/0000004/`. Allow approximately 61 GB
for this FP32 model-only output, in addition to space for the input checkpoint
and toy media. The example does not save optimizer state.

## Load the trained weights

```bash
python inference.py \
  --checkpoint "$PIXELUMM_TRAIN_OUTPUT/checkpoints/0000004" \
  --llm-path "$PIXELUMM_QWEN_DIR" \
  --task t2i --prompt "A red panda reading beside a window" \
  --height 256 --width 256 --seed 4396 \
  --output /absolute/path/to/pixelumm-results/toy-trained.png
```

For a CPU-only configuration check, run `python train_toy.py --help` or
`python train_toy.py ... --print-config` with the same paths and arguments.
