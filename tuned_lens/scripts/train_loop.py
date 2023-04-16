"""Training loop for training a TunedLens model against a transformer on a dataset."""
from argparse import Namespace
from collections import defaultdict
from typing import List
from pathlib import Path
from datasets import Dataset
from itertools import islice
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from tuned_lens import TunedLens
from tuned_lens.__main__ import Arg
from tuned_lens.residual_stream import ResidualStream
from tuned_lens.utils import (
    maybe_all_reduce,
    shift_labels,
    shift_preds,
    send_to_device,
)
import torch as th
import torch.distributed as dist

cli_args: List[Arg] = [
    {
        "name_or_flags": ["--constant"],
        "options": {"action": "store_true", "help": "Train only the bias term."},
    },
    {
        "name_or_flags": ["--extra-layers"],
        "options": {
            "type": int,
            "default": 0,
            "help": "Number of extra decoder layers.",
        },
    },
    {
        "name_or_flags": ["--lasso"],
        "options": {
            "type": float,
            "default": 0.0,
            "help": "LASSO (L1) regularization strength.",
        },
    },
    {
        "name_or_flags": ["--lens"],
        "options": {
            "type": Path,
            "help": "Directory containing a lens to warm-start training.",
        },
    },
    {
        "name_or_flags": ["--lr-scale"],
        "options": {
            "type": float,
            "default": 1.0,
            "help": "The default LR (1e-3 for Adam, 1.0 for SGD) is scaled by this "
                    "factor.",
        },
    },
    {
        "name_or_flags": ["--momentum"],
        "options": {
            "type": float,
            "default": 0.9,
            "help": "Momentum coefficient for SGD, or beta1 for Adam.",
        },
    },
    {
        "name_or_flags": ["--num-steps"],
        "options": {"type": int, "default": 250, "help": "Number of training steps."},
    },
    {
        "name_or_flags": ["--optimizer"],
        "options": {
            "type": str,
            "default": "sgd",
            "choices": ("adam", "sgd"),
            "help": "The type of optimizer to use.",
        },
    },
    {
        "name_or_flags": ["-o", "--output"],
        "options": {
            "type": Path,
            "required": True,
            "help": "File to save the lenses to. Defaults to the model name.",
        },
    },
    {
        "name_or_flags": ["--pre-ln"],
        "options": {
            "action": "store_true",
            "help": "Apply layer norm before, and not after, each probe.",
        },
    },
    {
        "name_or_flags": ["--resume"],
        "options": {"type": Path, "help": "File to resume training from."},
    },
    {
        "name_or_flags": ["--separate-unembeddings"],
        "options": {
            "action": "store_true",
            "help": "Learn a separate unembedding for each layer.",
        },
    },
    {
        "name_or_flags": ["--tokens-per-step"],
        "options": {
            "type": int,
            "default": 2**18,
            "help": "Number of tokens per step.",
        },
    },
    {
        "name_or_flags": ["--wandb"],
        "options": {"type": str, "help": "Name of run in Weights & Biases."},
    },
    {
        "name_or_flags": ["--warmup-steps"],
        "options": {
            "type": int,
            "default": None,
            "help": "Number of warmup steps. Defaults to min(0.1 * num_steps, 1000) "
                    "for Adam and 0 for SGD.",
        },
    },
    {
        "name_or_flags": ["--weight-decay"],
        "options": {
            "type": float,
            "default": 1e-3,
            "help": "Weight decay coefficient.",
        },
    },
    {
        "name_or_flags": ["--zero"],
        "options": {"action": "store_true", "help": "Use ZeroRedundancyOptimizer."},
    },
]


def train_loop(
    args: Namespace,
    model: th.nn.Module,
    data: Dataset,
    lens: TunedLens,
    nats_to_bpb: float,
):
    """Trains a TunedLens model against a transformer on a dataset.

    Args:
        args: The command-line arguments see __main__.py train subcommand.
        model: The transformer model to train.
        data: The dataset to train on.
        lens: The TunedLens model to train.
        nats_to_bpb: The ratio of nats to bits per byte for the dataset.
    """
    lens_size = sum(p.numel() * p.element_size() for p in lens.parameters())
    print(f"Tuned lens memory usage: {lens_size / 2 ** 20:.2f} MB per GPU")

    if args.constant:
        for probe in lens:
            probe.weight.requires_grad_(False)

    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        ddp_lens = DDP(lens, device_ids=[local_rank], find_unused_parameters=True)
    else:
        ddp_lens = lens

    dl = DataLoader(
        data.shuffle(seed=args.seed),  # type: ignore[arg-type]
        batch_size=args.per_gpu_batch_size,
    )
    if args.wandb and local_rank == 0:
        import wandb

        *_, name = args.model_name.split("/")
        wandb.init(
            config=vars(args),
            entity="eleutherai",
            group=name,
            name=args.wandb,
            project="tuned-lens",
        )
        wandb.watch(lens)

    # Don't train the unembedding matrix or final layer norm
    params = [p for p in ddp_lens.parameters() if p.requires_grad]

    β = args.momentum
    if args.optimizer == "sgd":
        config = dict(
            # PyTorch's implementation effectively scales the LR by 1 / (1 - β),
            # so we undo that here. See https://www.youtube.com/watch?v=k8fTYJPd3_I for
            # discussion. Once we do this, the optimal LR seems to be unity.
            lr=args.lr_scale * (1 - β),
            momentum=β,
            # Empirically Nesterov momentum seems to improve convergence speed.
            nesterov=True,
            weight_decay=args.weight_decay,
        )
        opt_class = th.optim.SGD
    elif args.optimizer == "adam":
        config = dict(
            # Helps convergence slightly by ensuring that the LR actually decays
            amsgrad=True,
            betas=(β, 0.999),
            lr=args.lr_scale * 1e-3,
            weight_decay=args.weight_decay,
        )
        opt_class = th.optim.Adam
    else:
        raise ValueError(f"Unknown optimizer '{args.optimizer}'")

    if args.zero:
        opt = ZeroRedundancyOptimizer(params, optimizer_class=opt_class, **config)
    else:
        opt = opt_class(params, **config)  # type: ignore[call-arg]

    if args.warmup_steps is None:
        # Adam generally performs poorly without an LR warmup
        if args.optimizer == "adam":
            args.warmup_steps = min(1000, args.num_steps // 5)
            print(f"Using {args.warmup_steps} LR warmup steps for Adam")
        else:
            args.warmup_steps = 0

    scheduler = get_linear_schedule_with_warmup(
        opt, args.warmup_steps, args.num_steps - args.warmup_steps
    )
    if args.resume:
        assert args.resume.is_dir()

        print(f"Loading checkpoint from {args.resume}")
        ddp_lens.load_state_dict(th.load(args.resume))

    # chunk_and_tokenize ensures the samples are all the same length
    tokens_per_sample = len(data[0]["input_ids"])
    samples_per_step, rem = divmod(args.tokens_per_step, tokens_per_sample)
    if rem:
        raise ValueError(
            f"Number of tokens per step ({args.tokens_per_step:_}) must be divisible "
            f"by the number of tokens per sample ({tokens_per_sample})."
        )

    global_batch_size = args.per_gpu_batch_size * world_size
    grad_acc_steps, rem = divmod(samples_per_step, global_batch_size)
    if rem:
        # If the number of samples per step isn't divisible by the global batch size,
        # use ceil division and let the user know about it.
        grad_acc_steps += 1
        adjusted_count = grad_acc_steps * global_batch_size * tokens_per_sample
        print(
            f"Note: Increasing grad acc steps from {grad_acc_steps - 1} to "
            f"{grad_acc_steps} to maintain load balance across {world_size} GPUs."
        )
        print(
            f"Using {adjusted_count:_} tokens per training step "
            f"({args.tokens_per_step:_} requested)."
        )
    else:
        print(f"Gradient accumulation steps: {grad_acc_steps}")
        print(f"Using {args.tokens_per_step:_} tokens per training step.")

    metrics = defaultdict(list)
    total_batches = args.num_steps * grad_acc_steps

    pbar = tqdm(islice(dl, total_batches), desc="Training", total=total_batches)
    for batch_idx, batch in enumerate(pbar, start=1):
        assert isinstance(batch, dict)
        batch = send_to_device(batch, th.device(local_rank))
        with th.autocast("cuda"):
            output = model(**batch, output_hidden_states=True)

        final_logits = output.logits
        stream = ResidualStream(
            embeddings=output.hidden_states[0], layers=output.hidden_states[1:-1]
        )

        shift = args.token_shift
        if args.loss == "ce":
            labels = batch["input_ids"]

            # Predict the *next* token by default w/ cross entropy
            if shift is None:
                shift = 1
        elif args.loss == "kl":
            labels = final_logits.log_softmax(dim=-1)

            # Match the *current* token distribution by default
            if shift is None:
                shift = 0
        else:
            raise NotImplementedError(f"Unknown loss {args.loss}")

        labels = shift_labels(labels, shift)

        # We do this sequentially to save VRAM
        for i, (name, h) in enumerate(stream.items()):
            # bfloat16 has larger dynamic range than float16 and seems to be better for
            # computing log softmax & KL loss
            with th.autocast("cuda", dtype=th.bfloat16):
                logits = shift_preds(ddp_lens(h, idx=i), shift)

                if args.loss == "ce":
                    loss = th.nn.functional.cross_entropy(
                        logits.flatten(0, -2), labels.flatten()
                    )
                elif args.loss == "kl":
                    loss = th.sum(
                        labels.exp() * (labels - logits.log_softmax(-1)), dim=-1
                    ).mean()
                else:
                    raise NotImplementedError

                # Log the loss *before* LASSO regularization
                logging_loss = loss.detach()
                maybe_all_reduce(logging_loss)
                if local_rank == 0:
                    metrics[f"loss/{name}"].append(logging_loss)

                # Add sparsity regularizer
                if args.lasso:
                    param_vec = th.nn.utils.parameters_to_vector(lens[i].parameters())
                    loss += args.lasso * param_vec.abs().sum()

                scaled_loss = loss / grad_acc_steps

            scaled_loss.backward()

        step, rem = divmod(batch_idx, grad_acc_steps)
        if rem == 0:
            th.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=False)
            scheduler.step()

            if local_rank == 0 and args.wandb:
                import wandb

                log_dict = {
                    k: th.stack(v).mean() * nats_to_bpb for k, v in metrics.items()
                }

                # Log statistics about optimizer & probes
                for i, probe in enumerate(lens):
                    name = "input" if i == 0 else f"{i - 1}.ffn"
                    states = [opt.state[p] for p in probe.parameters()]

                    # Approximate the true grad norm using the optimizer's moving avg
                    corr = 1 - β**step
                    if args.optimizer == "sgd" and not args.zero:
                        log_dict["grad_norm/" + name] = th.cat(
                            [
                                # Undo PyTorch's scaling of the gradient by 1 / (1 - β)
                                (1 - β) * s["momentum_buffer"].flatten() / corr
                                for s in states
                            ]
                        ).norm()
                    elif args.optimizer == "adam" and not args.zero:
                        log_dict["grad_norm/" + name] = th.cat(
                            [s["exp_avg"].flatten() / corr for s in states]
                        ).norm()

                    if isinstance(probe, th.nn.Linear):
                        log_dict["bias_norm/" + name] = probe.bias.data.norm()
                        log_dict["weight_norm/" + name] = probe.weight.data.norm()

                metrics.clear()
                wandb.log(log_dict)

        # Make the problem strictly convex with projected gradient descent,
        # centering the affine transform at each step
        lens.normalize_()

    if local_rank == 0:
        print(f"Saving lens to {args.output}")
        lens.save(args.output)
