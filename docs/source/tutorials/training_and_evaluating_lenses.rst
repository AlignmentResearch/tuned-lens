.. _training-and-evaluating-lenses:
    How to train and evaluate lenses on the pile

==============================
Training and evaluating lenses
==============================

**Downloading the Dataset**

The experiments in the paper were run by training a lens on the validation set of the pile.

.. code-block:: console

   wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
   unzstd val.jsonl.zst
   wget https://the-eye.eu/public/AI/pile/test.jsonl.zst
   unzstd test.jsonl.zst


**Training a Lens**

This will train a tuned lens on gpt-2 with the default hyper parameters. The model will
be automatically be downloaded from huggingface hub and cached locally.


Note this will only use one GPU for training a tutorial on multi-gpu training is coming soon.

.. code-block:: console

   tuned-lens train gpt2 val.jsonl
       --dataset the_pile all \
       --split validation \
       --output ./gpt-2-lens

**Evaluating a Lens**

Once you have a lens file either by training it yourself or by downloading it. You
can run various evaluations on it using the provided evaluation command.

.. code-block:: console

   tuned-lens eval gpt2 test.jsonl --lens ./gpt-2-lens
       --dataset the_pile all \
       --split validation \
       --output lens_eval_results.json
