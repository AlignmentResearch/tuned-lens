.. _training-and-evaluating-lenses:
    How to train and evaluate lenses on the pile

##############################
Training and evaluating lenses
##############################

In this section, we will discuss some of the technical details of training and evaluating your own lenses. First, we will briefly discuss single GPU training and evaluation. Then we will dive into some of the more technical aspects of training a model.

+++++++++++++++++++++++
Downloading the Dataset
+++++++++++++++++++++++

Before we can start training, we will need to set up our dataset. The experiments in the paper were run on the `pythia models <https://github.com/EleutherAI/pythia>`_ by training thus we train our lenses on the validation set of the pile. Let's first go ahead and download the validation and test splits of the pile.

.. code-block:: console

   wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
   unzstd val.jsonl.zst
   wget https://the-eye.eu/public/AI/pile/test.jsonl.zst
   unzstd test.jsonl.zst

+++++++++++++++
Training a Lens
+++++++++++++++

This command will train a tuned lens on `https://github.com/EleutherAI/pythia` with the default hyperparameters. The model will be automatically downloaded from the Hugging Face Hub and cached locally. You can adjust the per GPU batch size to maximize your GPU utilization.

.. code-block:: console

   python -m tuned_lens train \
        --model.name EleutherAI/pythia-160m-deduped \
        --data.name val.jsonl \
        --per_gpu_batch_size=1 \
        --output my_lenses/EleutherAI/pythia-160m-deduped

Once training is completed, this should save the trained lens to the `trained-lenses/pythia-160m-deduped` directory.

+++++++++++++++++
Evaluating a Lens
+++++++++++++++++

Once you have a lens trained, either by training it yourself, or by loading it from the hub, you can run various evaluations on it using the provided evaluation command.

.. code-block:: console

   python -m tuned_lens eval \
        --data.name test.jsonl \
        --model.name EleutherAI/pythia-160m-deduped \
        --tokens 16400000 \
        --lens_name my_lenses/EleutherAI/pythia-160m-deduped \
        --output evaluation/EleutherAI/pythia-160m-deduped

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
    --model.name EleutherAI/pythia-160m-deduped \
    --data.name val.jsonl \
    --per_gpu_batch_size=1 \
    --output my_lenses/EleutherAI/pythia-160m-deduped

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
    --model.name EleutherAI/pythia-160m-deduped \
    --data.name val.jsonl \
    --per_gpu_batch_size=1 \
    --output my_lenses/EleutherAI/pythia-160m-deduped \
    --fsdp

You can also use cpu offloading to train lenses on very large models while using less VRAM it can be enabled with the ``--cpu_offload`` flag. However, this substantially slows down training and is still experimental.

+++++++++++++++++++++++++++++
Checkpoint Resume
+++++++++++++++++++++++++++++

If you are running on a cluster with preemption you may want to be able to run a run with checkpoint resume. This can be enabled by passing the `--checkpoint_freq` flag with a number of steps between checkpoints.
By default checkpoints are saved to ``<output>/checkpoints`` this can be overridden with the ``--checkpoint_dir`` flag. There is a known issue with combining this with the zero optimizer, see [this issue](https://github.com/AlignmentResearch/tuned-lens/issues/96).

If checkpoints are present in the checkpoints dir, the trainer will automatically resume from the latest one.

++++++++++++++++++++++++++++++++++
Loading the Model Weights in int8
++++++++++++++++++++++++++++++++++

The `--precision int8` flag can be used to load the model's weights in a quantized int8 format. The `bitsandbytes` library must be installed for this to work. This should reduce VRAM usage by roughly a factor of two relative to float16 precision. Unfortunately, this option cannot be combined with `--fsdp` or `--cpu_offload`.

++++++++++++++++++++++++
Weights & Biases Logging
++++++++++++++++++++++++

To enable logging to ``wandb``, you can pass the ``--wandb <name-of-run>`` flag. This will log the training and evaluation metrics to ``wandb``. You will need to set the ``WANDB_API_KEY``, ``WANDB_ENTITY`` and ``WANDB_PROJECT`` environment variables in your environment. You can find your API key on your `wandb profile page <https://wandb.ai/settings>`_. To make this easy, you can create a ``.env`` file in the root of the project with the following contents.

.. code-block:: bash

    # .env
    WANDB_API_KEY= # your-api-key
    WANDB_ENTITY= # your-entity
    WANDB_PROJECT= # your-project-name

Then you can source it when you start your shell by running ``source .env``. For additional ``wandb`` environment variables, `see here <https://docs.wandb.ai/guides/track/advanced/environment-variables>`_.

++++++++++++++++++++
Uploading to the Hub
++++++++++++++++++++

Once you have trained a lens for a new model if you are feeling generous you can upload it to `our hugging face hub space <https://huggingface.co/spaces/AlignmentResearch/tuned-lens>`_ and share it with the world.

To do this first create a pull request on `the community tab <https://huggingface.co/spaces/AlignmentResearch/tuned-lens/discussions>`_.

Follow the commands to clone the repo and checkout your pr branch.

.. warning::
    Hugging face hub uses git-lfs to store large files. As a result you should generally work with `GIT_LFS_SKIP_SMUDGE=1` set when running `git clone` and `git checkout` commands.

Once you have checked out your branch you're branch copy the `config.json` and  `params.pt` produced by the training run to lens/<model-name> in the repo. Then add and commit the changes.

.. note::
    You shouldn't have to use `GIT_LFS_SKIP_SMUDGE=1` when adding and committing files.

Finally, in your pr description include the following information:
* The model name
* The dataset used to train the lens
* The training command used to train the lens
* And ideally, a link to the wandb run

We will review your pr and merge you're lens into the space. Thank you for contributing!
