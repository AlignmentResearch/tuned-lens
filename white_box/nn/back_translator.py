from .decoder import Decoder
import torch as th


class BackTranslator(th.nn.Module):
    def __init__(self, decoder: Decoder, lens: th.nn.Linear, h0: th.Tensor):
        super().__init__()

        self.decoder = decoder
        self.lens = lens
        self.h0 = h0

    def forward(self, h: th.Tensor):
        logits = self.decoder(h + self.lens(h))
        return self.decoder.invert(logits, h0=self.h0, lens=self.lens)


class BackTranslationWrapper(th.nn.Module):
    """Wraps a Huggingface transformers layer to add back-translation."""

    def __init__(self, layer: th.nn.Module, back_translator: BackTranslator):
        super().__init__()
        self.layer = layer
        self.back_translator = back_translator

    def forward(self, *args, **kwargs):
        h, *extras = self.layer(*args, **kwargs)
        return self.back_translator(h), *extras
