from transformers.models.bloom.modeling_bloom import BloomBlock
import torch as th


def instantiate_layer(model_config, layer_idx: int, model_type: str) -> th.nn.Module:
    if model_type == "bloom":
        from transformers.models.bloom.modeling_bloom import BloomBlock

        return _BloomBlockWrapper(BloomBlock(model_config))  # type: ignore[arg-type]
    if model_type == "gpt_neo":
        from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock

        return GPTNeoBlock(model_config, layer_idx)
    if model_type == "gpt_neox":
        from transformers.models.gpt_neox.modeling_gpt_neox import (
            GPTNeoXLayer,
        )

        return GPTNeoXLayer(model_config)  # type: ignore[arg-type]
    if model_type == "gpt2":
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        return GPT2Block(model_config, layer_idx)  # type: ignore[arg-type]
    if model_type == "opt":
        from transformers.models.opt.modeling_opt import OPTDecoderLayer

        return OPTDecoderLayer(model_config)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown model type '{model_type}'")


def maybe_wrap(layer: th.nn.Module) -> th.nn.Module:
    return _BloomBlockWrapper(layer) if isinstance(layer, BloomBlock) else layer


# Very annoying that we have to do this. See https://bit.ly/3XSQ7W6 for context on
# what we're doing here.
class _BloomBlockWrapper(th.nn.Module):
    def __init__(self, block: BloomBlock):
        super().__init__()
        self.block = block

    def forward(self, x: th.Tensor) -> th.Tensor:
        from transformers.models.bloom.modeling_bloom import (
            BloomModel,
            build_alibi_tensor,
        )

        batch_size, seq_len, _ = x.shape
        dummy_mask = x.new_ones([batch_size, seq_len])

        # Causal mask isn't created inside the block itself, so we have to do it here.
        # Weirdly _prepare_attn_mask doesn't depend on `self` at all but is still an
        # instance method for some reason, so we pass `None` as the first argument.
        causal_mask = BloomModel._prepare_attn_mask(
            None, dummy_mask, (batch_size, seq_len), 0  # type: ignore[arg-type]
        )
        alibi = build_alibi_tensor(dummy_mask, self.block.num_heads, x.dtype)
        h, *_ = self.block(x, alibi, causal_mask)
        return h
