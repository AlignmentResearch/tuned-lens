.. _training-and-evaluating-lenses:
    How to train and evaluate lenses on the pile


Downloading the datasets
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
   unzstd val.jsonl.zst

   wget https://the-eye.eu/public/AI/pile/test.jsonl.zst
   unzstd test.jsonl.zst

Evaluating a Lens
~~~~~~~~~~~~~~~~~

Once you have a lens file either by training it yourself of by downloading it. You
can run various evaluations on it using the provided evaluation command.

.. code-block:: console

   tuned-lens eval gpt2 test.jsonl --lens gpt-2-lens
       --dataset the_pile all \
       --split validation \
       --output lens_eval_results.json

Training a Lens
~~~~~~~~~~~~~~~

This will train a tuned lens on gpt-2 with the default hyper parameters.

.. code-block:: bash

   tuned-lens train gpt2 val.jsonl
       --dataset the_pile all \
       --split validation \
       --output gpt-2-lens
