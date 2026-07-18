# Qwen Model Support

This note documents the Qwen support contribution for
`AlignmentResearch/tuned-lens`.

## Before

`tuned_lens.model_surgery.get_final_norm()` and
`tuned_lens.model_surgery.get_transformer_layers()` only recognized model families
such as OPT, GPT-NeoX, GPT-2, LLaMA, Mistral, and Gemma. Qwen models reached the
fallback branch and raised:

```text
NotImplementedError: Unknown model type <class 'transformers.models.qwen3.modeling_qwen3.Qwen3Model'>
```

That blocked even LogitLens-style prediction trajectories for Qwen3.

## Changes

- Added conditional Qwen model detection for `Qwen2Model` and `Qwen3Model`.
- Mapped Qwen final normalization to `base_model.norm`.
- Mapped Qwen decoder layers to `base_model.layers`.
- Added tiny random Qwen2 and Qwen3 configs to the existing random model test
  fixture, so model surgery and lens construction are tested without downloading
  large checkpoints.

The Qwen checks are conditional on the installed Transformers version. Older
Transformers versions can still import `tuned_lens`; newer versions that expose
Qwen2/Qwen3 classes get support automatically.

## Real Pretrained Result

The snapshot below was generated with the real pretrained
`Qwen/Qwen3-0.6B` checkpoint, not a randomly initialized mock model. This is a
short LogitLens integration result: it verifies that tuned-lens can locate the
Qwen final norm and decoder layers, then build a `PredictionTrajectory` from the
model's real hidden states.

![Qwen3-0.6B LogitLens result](qwen3_0_6b_logit_lens_result.png)

Run details:

- Model: `Qwen/Qwen3-0.6B`
- Prompt: `AI alignment research helps language models`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- Dtype: `float16`
- Peak CUDA memory: `1.41 GB`
- Layer path returned by model surgery: `base_model.layers`
- Layers found: `28`
- Final norm type: `Qwen3RMSNorm`

| Layer | Top token at final prompt position | Top probability | Mean entropy | Mean forward KL |
| --- | ---: | ---: | ---: | ---: |
| 0 | ` models` (4119) | 1.0000 | 0.0000 | 102.8691 |
| 1 | ` model` (1614) | 0.2509 | 2.8800 | 11.4868 |
| 14 | `erve` (5852) | 0.0979 | 5.2927 | 5.6683 |
| 27 | ` to` (311) | 0.2655 | 4.8768 | 0.4478 |
| output | ` to` (311) | 0.2135 | 4.8618 | 0.0000 |

This is not a trained tuned-lens benchmark. It is a real pretrained-model
compatibility and LogitLens trajectory result for Qwen3.

Validation command:

```bash
python -m pytest tests/test_model_surgery.py tests/test_lenses.py::test_tuned_lens_from_model -q -k qwen
```

Result:

```text
6 passed, 22 deselected
```
