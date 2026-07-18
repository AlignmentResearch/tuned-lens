# Qwen Model Support

This note documents my Qwen support contribution for `AlignmentResearch/tuned-lens`.

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

- Added Qwen-family decoder model detection based on `model_type`, `layers`, and
  `norm`.
- Mapped Qwen final normalization to `base_model.norm`.
- Mapped Qwen decoder layers to `base_model.layers`.
- Added tiny random Qwen2, Qwen2-MoE, Qwen3, Qwen3-MoE, and Qwen3-Next configs
  to the existing random model test fixture, so model surgery and lens
  construction are tested without downloading large checkpoints.

The Qwen check is structural, so it also covers versioned Qwen models that share
these decoder architectures, such as Qwen2.5-style models exposed as `qwen2` by
Transformers. Older Transformers versions can still import `tuned_lens`; newer
versions get support automatically when the corresponding Qwen configs exist.

## Real Pretrained Result

I ran a real pretrained `Qwen/Qwen3-0.6B` LogitLens check. The result confirms
that tuned-lens can locate the Qwen final norm and decoder layers and build a
`PredictionTrajectory` from real model hidden states.

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

This is a real pretrained-model compatibility result for Qwen3. It is not a
trained tuned-lens benchmark.

Validation command:

```bash
python -m pytest tests/test_model_surgery.py tests/test_lenses.py -q -k qwen
```

Result:

```text
20 passed, 33 deselected
```
