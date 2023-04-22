.. _loading-pertained-lenses:
    How to train and evaluate lenses on the pile

==========================
Loading a pre-trained lens
==========================

**From the hugging face API**

.. _pre-trained lenses folder: https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens

First check if there is a pre-trained lens available in our spaces' `pre-trained lenses folder`_.

Once you have found a lens that you want to use, you can load it using the simply load it
and its corresponding tokenizer using the hugging face API. A tuned lens is always associated with
a model that was used to train it so first load the model and then the lens.

>>> import torch
>>> from tuned_lens import TunedLens
>>> from transformers import AutoModelForCausalLM
>>> device = torch.device('cpu')
>>> model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m-deduped')
>>> tuned_lens = TunedLens.from_pretrained("pythia-160m-deduped", model=model, map_location=device)

If you want to load from your own code space you can override the default
by providing the correct environment variables see :ref:`tuned\_lens.load\_artifacts`.

**From the a local folder**

If you have trained a lens and want to load it for inference simply pass the
model used to train it and the folder you saved it to.

.. testsetup::

    from tuned_lens import TunedLens
    from tempfile import TemporaryDirectory
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m-deduped')
    temp_dir = TemporaryDirectory()
    directory_path = temp_dir.name

.. doctest::

    >>> lens = TunedLens.from_model(model)
    >>> # Do some thing
    >>> lens.save(directory_path)
    >>> lens = TunedLens.from_pretrained(directory_path, model=model)

.. testcleanup::

    temp_dir.cleanup()

Note the folder structure must look as follows:

.. code-block:: text

    path/to/folder
    ├── config.json
    └── params.pt

If you saved the model using ``tuned_lens.save("path/to/folder")`` then this should already be the case.
