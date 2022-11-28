from argparse import Namespace
from collections import defaultdict
from datasets import Dataset
from itertools import islice
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from white_box import ResidualStats, ResidualStream, TunedLens
from white_box.utils import (
    maybe_all_reduce,
    maybe_shift_labels,
    maybe_shift_preds,
    send_to_device,
)
import torch as th
import torch.distributed as dist


def train_loop(
    args: Namespace,
    model: th.nn.Module,
    data: Dataset,
    lens: TunedLens,
    nats_to_bpb: float,
):
    lens_size = sum(p.numel() * p.element_size() for p in lens.parameters())
    print(f"Tuned lens memory usage: {lens_size / 2 ** 20:.2f} MB per GPU")

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

    # Running mean & covariance of the hidden states
    first_token_stats = ResidualStats()
    stream_stats = ResidualStats()

    if args.wandb and local_rank == 0:
        import wandb

        wandb.init(
            name=args.wandb, project=args.model_name.split("/")[-1], config=vars(args)
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
            args.warmup_steps = min(1000, args.num_steps // 10)
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

    print(f"Using {args.tokens_per_step:_} tokens per training step.")

    # TODO: Make this do the right thing when there's a remainder
    global_batch_size = args.per_gpu_batch_size * world_size
    grad_acc_steps = samples_per_step // global_batch_size

    metrics = defaultdict(list)
    total_batches = args.num_steps * grad_acc_steps
    print(f"Gradient accumulation steps: {grad_acc_steps}")

    pbar = tqdm(islice(dl, total_batches), desc="Training", total=total_batches)
    for batch_idx, batch in enumerate(pbar, start=1):
        assert isinstance(batch, dict)
        batch = send_to_device(batch, th.device(local_rank))
        output = model(**batch, output_hidden_states=True)

        final_logits = output.logits
        stream = ResidualStream(
            embeddings=output.hidden_states[0], layers=output.hidden_states[1:-1]
        )
        if args.residual_stats:
            first_tokens = stream.map(lambda x: x[:, 0])
            rest = stream.map(lambda x: x[:, 1:])

            first_token_stats.update(first_tokens)
            stream_stats.update(rest)

        shift = args.token_shift
        if args.loss == "ce":
            labels = batch["input_ids"]

            # Predict the *next* token by default w/ cross entropy
            if shift is None:
                shift = 1
        elif args.loss in ("kl", "kl-reverse"):
            labels = final_logits.log_softmax(dim=-1)

            # Match the *current* token distribution by default
            if shift is None:
                shift = 0
        else:
            raise NotImplementedError(f"Unknown loss {args.loss}")

        labels = maybe_shift_labels(labels, shift)

        # We do this sequentially to save VRAM
        for i, (name, h) in enumerate(stream.items()):
            # bfloat16 has larger dynamic range than float16 and seems to be better for
            # computing log softmax & KL loss
            with th.autocast("cuda", dtype=th.bfloat16):
                logits = maybe_shift_preds(ddp_lens(h, idx=i), shift)

                if args.loss == "ce":
                    loss = th.nn.functional.cross_entropy(
                        logits.flatten(0, -2), labels.flatten()
                    )

                # KL(P || Q)
                elif args.loss == "kl":
                    loss = th.sum(
                        labels.exp() * (labels - logits.log_softmax(-1)), dim=-1
                    ).mean()

                # KL(Q || P)
                elif args.loss == "kl-reverse":
                    log_probs = logits.log_softmax(-1)
                    loss = th.sum(
                        log_probs.exp() * (log_probs.log_softmax(-1) - labels), dim=-1
                    ).mean()
                else:
                    raise NotImplementedError

                # Log the loss *before* LASSO regularization
                logging_loss = loss.detach()
                maybe_all_reduce(logging_loss, "mean")
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
            opt.zero_grad()
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
                    if args.optimizer == "sgd":
                        log_dict["grad_norm/" + name] = th.cat(
                            [
                                # Undo PyTorch's scaling of the gradient by 1 / (1 - β)
                                (1 - β) * s["momentum_buffer"].flatten() / corr
                                for s in states
                            ]
                        ).norm()
                    elif args.optimizer == "adam":
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

    if args.residual_stats:
        first_token_stats.all_reduce_()
        stream_stats.all_reduce_()
        if local_rank == 0:
            th.save(first_token_stats, args.output / "first_token_stats.pt")
            th.save(stream_stats, args.output / "stream_stats.pt")
