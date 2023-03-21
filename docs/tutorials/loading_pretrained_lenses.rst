.. _loading-pertained-lenses:
    How to train and evaluate lenses on the pile

==========================
Loading a pre-trained lens
==========================

**From the hugging face API**

.. _pre-trained lenses folder: https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens

First check if there is a pre-trained lens available in our spaces' `pre-trained lenses folder`_.

Once you have found a lens that you want to use, you can load it using the simply load it
and its corresponding tokenizer using the hugging face API.


>>> import torch
>>> from tuned_lens import TunedLens
>>> from transformers import AutoModelForCausalLM
>>> device = torch.device('cpu')
>>> model = AutoModelForCausalLM.from_pretrained('gpt2-large')
>>> tuned_lens = TunedLens.load("gpt2-large", map_location=device)

If you want to load from your own code space you can override the default
by providing the correct environment variables see :ref:`tuned\_lens.load\_artifacts`.

**From the a local folder**

If you have trained a lens and want to load it for inference simply pass the folder
to the load method.

>>> tuned_lens = TunedLens.load("path/to/folder")

Note the folder structure must look as follows:
.. code-block:: text
    path/to/folder
    ├── config.json
    └── params.pt

If you saved the model using ``tuned_lens.save("path/to/folder")`` then this should already be the case.
