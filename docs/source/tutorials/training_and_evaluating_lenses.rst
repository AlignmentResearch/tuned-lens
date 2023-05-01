.. _training-and-evaluating-lenses:
    How to train and evaluate lenses on the pile

##############################
Training and evaluating lenses
##############################

In this section, we will discuss some of the technical details of training and evaluating your own lenses. First, we will briefly discuss single GPU training and evaluation. Then we will dive into some of the more technical aspects of training a model.

+++++++++++++++++++++++
Downloading the Dataset
+++++++++++++++++++++++

Before we can start training, we will need to set up our dataset. The experiments in the paper were run by training a lens on the validation set of the pile. Let's first go ahead and download the validation and test splits of the pile.

.. code-block:: console

   wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
   unzstd val.jsonl.zst
   wget https://the-eye.eu/public/AI/pile/test.jsonl.zst
   unzstd test.jsonl.zst

+++++++++++++++
Training a Lens
+++++++++++++++

This command will train a tuned lens on GPT-2 with the default hyperparameters. The model will be automatically downloaded from the Hugging Face Hub and cached locally. You can adjust the per GPU batch size to maximize your GPU utilization.

.. code-block:: console

   python -m tuned_lens train \
        --model.name gpt2 \
        --data.name val.jsonl \
        --per_gpu_batch_size=1

Once training is completed, this should save the trained lens to the `gpt2` directory. You can specify a different directory by passing the `--output` flag.

+++++++++++++++++
Evaluating a Lens
+++++++++++++++++

Once you have a lens file, either by training it yourself or by downloading it, you can run various evaluations on it using the provided evaluation command.

.. code-block:: console

   python -m tuned_lens eval \
        --model.name gpt2 \
        --data.name test.jsonl \
        --per_gpu_batch_size=1

++++++++++++++++++++++++++++++++++++++++++++
Distributed Data Parallel Multi-GPU Training
++++++++++++++++++++++++++++++++++++++++++++

You can also use `torch elastic launch <https://pytorch.org/docs/stable/elastic/run.html>`_ to do multi-GPU training. This will default to doing `distributed data parallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ training for the lens. Note
that this still requires the transformer model itself to fit on a single GPU. However, since we are almost always using some form of gradient accumulation, this usually still speeds up training significantly.

.. code-block:: console

    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=<num_gpus> \
    -m tuned_lens train \
    --model.name gpt2 \
    --data.name val.jsonl \
    --per_gpu_batch_size=1

++++++++++++++++++++++++++++++++++++++++++++++
Fully Sharded Data Parallel Multi-GPU Training
++++++++++++++++++++++++++++++++++++++++++++++

If the transformer model does not fit on a single GPU, you can also use `fully sharded data parallel <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_ training. Note that the lens is still trained using DDP, only the transformer itself is sharded. To enable this, you can pass the `--fsdp` flag.

.. code-block:: console

    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=<num_gpus> \
    -m tuned_lens train \
    --model.name gpt2 \
    --data.name val.jsonl \
    --per_gpu_batch_size=1 \
    --fsdp

You can also use cpu offloading to train lenses on very large models while using less VRAM it can be enabled with the `--cpu_offload` flag. However, this substantially slows down training and is still experimental.

++++++++++++++++++++++++
Weights & Biases Logging
++++++++++++++++++++++++

To enable logging to `wandb`, you can pass the `--wandb <name-of-run>` flag. This will log the training and evaluation metrics to Wandb. You will need to set the `WANDB_API_KEY`, `WANDB_ENTITY` and `WANDB_PROJECT`` environment variables in your environment. You can find your API key on your `wandb profile page <https://wandb.ai/settings>`_. To make this easy, you can create a `.env`` file in the root of the project with the following contents.

.. code-block:: bash

    # .env
    WANDB_API_KEY= # your-api-key
    WANDB_ENTITY= # your-entity
    WANDB_PROJECT= # your-project-name

Then you can source it when you start your shell by running `source .env`. For additional Wandb environment variables, `see here <https://docs.wandb.ai/guides/track/advanced/environment-variables>`_.
