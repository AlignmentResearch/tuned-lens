.. include:: ../../README.md
   :parser: myst_parser.sphinx_

API Reference
~~~~~~~~~~~~~

.. autosummary::
    :toctree: _api
    :caption: API Reference
    :template: autosummary/module.rst
    :recursive:

    tuned_lens.nn.lenses
    tuned_lens.nn.unembed
    tuned_lens.plotting
    tuned_lens.load_artifacts

.. toctree::
    :maxdepth: 2
    :caption: Tutorials
    :hidden:

    tutorials/loading_pretrained_lenses.rst
    tutorials/training_and_evaluating_lenses.rst
    tutorials/prediction_trajectories.ipynb
    tutorials/combining_with_transformer_lens.ipynb
    tutorials/maintainers_guide.rst
