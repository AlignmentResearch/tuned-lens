
Tuned Lens
~~~~~~~~~~
**The tuned-lens package provides tools for training, evaluating and tunning inference
on tuned lens models on transformer-based language models.** A tuned lens allows us
to peak at the iterative computations a transformer uses to compute the next token.

==================
What is a Lens?
==================
A lens into a transformer with n layers allows you to replace the last $m$ layers of the model with an [affine transformation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (we call these affine translators).

This skips over these last few layers and lets you see the best prediction that can be made from the model's intermediate representations, i.e. the residual stream, at layer $n - m$. Since the representations may be rotated, shifted, or stretched from layer to layer it's useful to train an affine specifically on each layer. This training is what differentiates this method from simpler approaches that decode the residual stream of the network directly using the unembeding layer i.e. the [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). We explain this process and its applications in a forthcoming paper "Eliciting Latent Predictions from Transformers with the Tuned Lens".

Tutorials
~~~~~~~~~
.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/training_and_evaluating_lenses.rst
   tutorials/loading_pretrained_lenses.rst

API Reference
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _api
   :caption: API Reference
   :template: autosummary/module.rst
   :recursive:

   tuned_lens.nn.lenses
   tuned_lens.nn.decoder
   tuned_lens.plotting.plot_lens
