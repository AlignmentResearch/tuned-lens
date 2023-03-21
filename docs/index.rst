==========
Tuned Lens
==========
**The tuned-lens package provides tools for training, evaluating and tunning inference
on tuned lens models on transformer-based language models.** A tuned lens allows us
to peak at the iterative computations a transformer uses to compute the next token.

.. _`Eliciting Latent Predictions from Transformers with the Tuned Lens`: https://arxiv.org/abs/2303.08112
.. _`affine transformation`: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
.. _`logit lens`: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

What is a Lens?
~~~~~~~~~~~~~~~
A lens into a transformer with n layers allows you to replace the last $m$ layers of the model with an `affine transformation`_ (we call these affine translators).

This skips over these last few layers and lets you see the best prediction that can be made from the model's intermediate representations, i.e. the residual stream, at layer $n - m$. Since the representations may be rotated, shifted, or stretched from layer to layer it's useful to train an affine specifically on each layer. This training is what differentiates this method from simpler approaches that decode the residual stream of the network directly using the unembeding layer i.e. the `logit lens`_. We explain this process and its applications in the paper `Eliciting Latent Predictions from Transformers with the Tuned Lens`_.

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
    tuned_lens.load_artifacts


Citation
~~~~~~~~
.. code-block:: text

    @misc{belrose2023eliciting,
        title={Eliciting Latent Predictions from Transformers with the Tuned Lens},
       author={Nora Belrose and Zach Furman and Logan Smith and Danny Halawi and Igor Ostrovsky and Lev McKinney and Stella Biderman and Jacob Steinhardt},
        year={2023},
        eprint={2303.08112},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
