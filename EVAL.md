# Optional evaluation

The repository includes qualitative image/video generation and two VLM
benchmark runners. These tools are separate from the inference and four-step
toy training quick starts.

| Evaluation | Entrypoint |
| --- | --- |
| Image/video generation | `train/eval_utils.py` |
| Image understanding (Regular21) | `eval/vlm/core13/` |
| Video understanding (Video4) | `eval/vlm/video_4/` |

Install the optional evaluation dependencies with
`python -m pip install -r requirements-eval.txt`, then check the F22
checkpoint:

```bash
python -m eval.vlm.preflight --checkpoint "$PIXELUMM_CKPT"
```

The VLM runners use the pinned `lmms-eval` installation in the active Python
environment. Point `--lmms-root` at that package:

```bash
PIXELUMM_LMMS_ROOT="$(python -c 'import lmms_eval; from pathlib import Path; print(Path(lmms_eval.__file__).parent.parent)')"
python -m eval.vlm.core13.run \
  --checkpoint "$PIXELUMM_CKPT" --lmms-root "$PIXELUMM_LMMS_ROOT"
python -m eval.vlm.video_4.run \
  --checkpoint "$PIXELUMM_CKPT" --lmms-root "$PIXELUMM_LMMS_ROOT"
```

The benchmark datasets are not included in this repository. The runners use
the usual Hugging Face cache by default; set `PIXELUMM_VLM_HF_HOME` to another
populated cache when needed. See the READMEs under `eval/vlm/` for task lists
and runner options. Store benchmark outputs outside the source checkout.
