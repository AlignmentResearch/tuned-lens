.. _training-and-evaluating-lenses:
    How to train and evaluate lenses on the pile

==============================
Training and evaluating lenses
==============================

Here we are going to discuss the some of the technical details of training and evaluating you own lenses. First we will briefly discuss single GPU training and evaluation. Then we will dive into some of the more technical aspects of training a model.


**Downloading the Dataset**


However, before we can start training we will need to get our dataset setup. The experiments in the paper were run by training a lens on the validation set of the pile. Let's first go ahead and download the validation and test splits of the pile.

.. code-block:: console

   wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
   unzstd val.jsonl.zst
   wget https://the-eye.eu/public/AI/pile/test.jsonl.zst
   unzstd test.jsonl.zst


**Training a Lens**

This will train a tuned lens on gpt-2 with the default hyper parameters. The model will be automatically be downloaded from huggingface hub and cached locally. You can adjust the per gpu batch size to maximize you gpu utilization.


.. code-block:: console

   python -m tuned_lens train \
        --model.name gpt2 \
        --data.name val.jsonl \
        --per_gpu_batch_size=1

Once training is completed this should save the trained lens to the `gpt2` directory.
You can specify a different directory by passing the `--output` flag.

**Evaluating a Lens**

Once you have a lens file either by training it yourself or by downloading it. You
can run various evaluations on it using the provided evaluation command.

.. code-block:: console

   python -m tuned_lens eval \
        --model.name gpt2 \
        --data.name test.jsonl \
        --per_gpu_batch_size=1

**Distributed Data Parallel Multi-GPU Training**

You can also use `torches elastic launch<https://pytorch.org/docs/stable/elastic/run.html>`_ to do multi gpu training. This will default to doing `distributed data parallel<https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ training. This will requite you cannot fit the model and lenses on a single gpu. Since we are almost always using some form of gradient accumulation this usually speeds up training significantly.

.. code-block:: console
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=<num_gpus> \
    -m tuned_lens train \
    --model.name gpt2 \
    --data.name val.jsonl \
    --per_gpu_batch_size=1

**Fully Sharded Data Parallel Multi-GPU Training**

If the model does not fit on a single gpu you can also use `fully sharded data parallel<https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_ training. Note the lens is still trained using DDP only the model is sharded. This is useful for very large models that do not fit on a single gpu. To enable this you can pass the `--fsdp` flag.

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


Note support for cpu-offloading is currently experimental. This substantially slows training but allows for very large models to be
run using less vram.

**Wandb Logging**
To enable logging to wandb you can pass the `--wandb <name-of-run>` flag. This will log the training and evaluation metrics to wandb. You will need to set the `WANDB_API_KEY`, `WANDB_PROJECT` environment variables present in your environment. You can find your api key on your `wandb profile page<https://wandb.ai/profile>`_. To make this easy you can create .env file in the root of the project with the following contents.

```
# .env
WANDB_API_KEY=<your-api-key>
WANDB_ENTITY=<>
WANDB_PROJECT=<your-project-name>
```

Then you can source it when you start your shell. For additional wandb environment variables see `here<https://docs.wandb.ai/guides/track/advanced/environment-variables>`_.
