"""Train or evaluate a tuned lens for a language model."""

from argparse import Namespace
from collections import defaultdict
from contextlib import nullcontext, redirect_stdout
from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
from itertools import islice
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from argparsers import get_lens_parser
from white_box import ResidualStats, ResidualStream, TunedLens
from white_box.data import chunk_and_tokenize, silence_datasets_messages
from white_box.model_surgery import get_transformer_layers
from white_box.utils import send_to_device
import json
import torch as th
import torch.distributed as dist


dist.init_process_group("nccl")
local_rank = dist.get_rank()


def main(args):
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    silence_datasets_messages()

    print(f"Loading dataset '{' '.join(args.dataset)}'")
    if len(args.dataset) == 1 and args.dataset[0].endswith(".jsonl"):
        dataset = Dataset.from_json(args.dataset[0])
        assert isinstance(dataset, Dataset)
    else:
        dataset = load_dataset(*args.dataset, split=args.split)
        if not isinstance(dataset, (Dataset, DatasetDict)):
            raise ValueError("Only Dataset and DatasetDict instances are supported.")

    processed = chunk_and_tokenize(dataset, tokenizer, text_key=args.text_column)
    assert isinstance(processed, Dataset)
    processed = processed.shard(dist.get_world_size(), local_rank)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
    )
    model.eval()
    model.requires_grad_(False)
    assert isinstance(model, PreTrainedModel)

    th.cuda.set_device(local_rank)

    # Can be set either in eval or in training; in eval it's required
    if args.lens:
        lens = TunedLens.load(args.lens).cuda(local_rank)
    else:
        lens = TunedLens(
            model,
            include_final=args.train_final_lens,
            orthogonal=args.orthogonal,
            rank=args.rank,
            sublayers=args.sublayers,
        ).cuda(local_rank)

    if args.fsdp:
        _, layers = get_transformer_layers(model)
        layer_cls = type(layers[0])
        print(f"Using '{layer_cls.__name__}' for transformer_auto_wrap_policy.")

        model = FSDP(
            model,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
            ),
            cpu_offload=CPUOffload(offload_params=True),
            device_id=local_rank,
            mixed_precision=MixedPrecision(
                param_dtype=th.float16,
                reduce_dtype=th.float16,
                buffer_dtype=th.float16,
            ),
        )
    else:
        model.to(local_rank)

    if args.command == "train":
        train_loop(args, model, processed, lens)
    elif args.command == "eval":
        eval_loop(args, model, processed, lens)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def maybe_shift_labels(x: th.Tensor, shift: int):
    if shift > 0:
        return x[:, shift:]
    if shift < 0:
        return x[:, :shift]

    return x


def maybe_shift_preds(x: th.Tensor, shift: int):
    if shift > 0:
        return x[:, :-shift]
    if shift < 0:
        return x[:, -shift:]

    return x


@th.autocast("cuda")
@th.no_grad()
def eval_loop(args: Namespace, model: th.nn.Module, data: Dataset, lens: TunedLens):
    local_rank = dist.get_rank()
    dl = DataLoader(data, batch_size=args.per_gpu_batch_size)  # type: ignore[arg-type]

    # Running mean & variance of the residuals
    residual_stats = ResidualStats()
    stream_stats = ResidualStats()

    # Keys are names of layers
    baseline_dict = defaultdict(list)
    ce_dict = defaultdict(list)
    kl_dict = defaultdict(list)
    final_losses = []

    for batch in tqdm(dl, desc="Evaluating", position=local_rank):
        batch = send_to_device(batch, th.device(local_rank))
        output = model(**batch, labels=batch["input_ids"], output_hidden_states=True)

        final_log_probs = output.logits.log_softmax(dim=-1)
        final_probs = final_log_probs.exp()

        final_losses.append(output.loss)
        labels = maybe_shift_labels(batch["input_ids"], 1).flatten()

        stream = ResidualStream(
            embeddings=output.hidden_states[0], layers=output.hidden_states[1:-1]
        )

        # Do this sequentially to save VRAM
        for i, (name, h) in enumerate(stream.items()):
            logits = lens(h, idx=i)
            log_probs = logits.log_softmax(dim=-1)

            baseline_dict[name].append(
                th.nn.functional.cross_entropy(
                    maybe_shift_preds(lens.decode(h), 1).flatten(0, 1), labels
                )
            )
            ce_dict[name].append(
                th.nn.functional.cross_entropy(
                    maybe_shift_preds(logits, 1).flatten(0, 1), labels
                )
            )
            kl_dict[name].append(
                th.sum(final_probs * (final_log_probs - log_probs), dim=-1).mean()
            )

        # Don't let the processes get too out of sync
        dist.barrier()
        if args.residual_stats:
            residual_stats.update(stream.residuals())
            stream_stats.update(stream)

    baseline_means = {k: th.stack(v).mean() for k, v in baseline_dict.items()}
    ce_means = {k: th.stack(v).mean() for k, v in ce_dict.items()}
    kl_means = {k: th.stack(v).mean() for k, v in kl_dict.items()}
    final_loss = th.stack(final_losses).mean()

    for v in baseline_means.values():
        dist.all_reduce(v)
        v /= dist.get_world_size()

    for v in ce_means.values():
        dist.all_reduce(v)
        v /= dist.get_world_size()

    for v in kl_means.values():
        dist.all_reduce(v)
        v /= dist.get_world_size()

    dist.all_reduce(final_loss)
    final_loss /= dist.get_world_size()

    if local_rank == 0:
        json_path = args.output or (args.lens / "eval.json")
        print(f"Saving results to '{json_path}'")

        with open(json_path, "w") as f:
            json.dump(
                {
                    "baseline": {k: v.item() for k, v in baseline_means.items()},
                    "ce": {k: v.item() for k, v in ce_means.items()},
                    "kl": {k: v.item() for k, v in kl_means.items()},
                    "final_loss": final_loss.item(),
                },
                f,
                indent=2,
            )

    if args.residual_stats:
        stream_stats.all_reduce_()
        residual_stats.all_reduce_()

        if local_rank == 0:
            th.save(stream_stats, args.output / "stream_stats.pt")
            th.save(residual_stats, args.output / "residual_stats.pt")


def train_loop(args: Namespace, model: th.nn.Module, data: Dataset, lens: TunedLens):
    local_rank = dist.get_rank()
    ddp_lens = DDP(lens, device_ids=[local_rank], find_unused_parameters=True)
    dl = DataLoader(
        data.shuffle(seed=args.seed),  # type: ignore[arg-type]
        batch_size=args.per_gpu_batch_size,
    )

    if args.wandb and local_rank == 0:
        import wandb

        wandb.init(
            name=args.wandb, project=args.model_name.split("/")[-1], config=vars(args)
        )
        wandb.watch(lens)

    # Don't train the unembedding matrix or final layer norm
    params = [p for p in ddp_lens.parameters() if p.requires_grad]

    β = args.momentum
    config = dict(
        # PyTorch's momentum implementation effectively scales the LR by 1 / (1 - β),
        # so we undo that here. See https://www.youtube.com/watch?v=k8fTYJPd3_I for
        # discussion. Interestingly, once we do this, the optimal LR seems to be unity.
        lr=args.lr * (1 - β),
        momentum=β,
        # Empirically Nesterov momentum seems to improve convergence speed.
        nesterov=True,
        # Training a lens is only weakly convex, with many near-zero eigenvalues in the
        # Hessian spectrum. Without weight decay, there's a tendency for less important
        # parameters to "drift" away from their zero initialization.
        weight_decay=args.weight_decay,
    )

    # It turns out to be pretty important to use SGD with momentum and not Adam. Since
    # we zero-initialize the probes, we start out with relatively small grad norms, and
    # Adam's adaptive learning rate bumps up the step size way too much.
    if args.zero:
        opt = ZeroRedundancyOptimizer(params, optimizer_class=th.optim.SGD, **config)
    else:
        opt = th.optim.SGD(params, **config)

    # Simple linear LR decay schedule
    scheduler = LambdaLR(opt, lambda t: 1 - t / args.num_steps)
    if args.resume:
        assert args.resume.is_dir()

        print(f"Loading checkpoint from {args.resume}")
        opt_path = args.resume / "optimizer.pt"
        ddp_lens.load_state_dict(th.load(args.resume))

        if opt_path.exists():
            print(f"Loading optimizer state from {opt_path}")
            opt.load_state_dict(th.load(opt_path))
        else:
            print("No optimizer state found. Starting Adam from scratch.")

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
    global_batch_size = args.per_gpu_batch_size * dist.get_world_size()
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

        labels = maybe_shift_labels(labels, shift)

        # We do this sequentially to save VRAM
        for i, (name, h) in enumerate(stream.items()):
            # bfloat16 has larger dynamic range than float16 and seems to be better for
            # computing log softmax & KL loss
            with th.autocast("cuda", dtype=th.bfloat16):
                preds = maybe_shift_preds(ddp_lens(h, idx=i), shift)

                if args.loss == "ce":
                    loss = th.nn.functional.cross_entropy(
                        preds.flatten(0, -2), labels.flatten()
                    )
                elif args.loss == "kl":
                    loss = th.sum(
                        labels.exp() * (labels - preds.log_softmax(-1)), dim=-1
                    ).mean()
                else:
                    raise NotImplementedError

                # Log the loss *before* LASSO regularization
                metrics[f"loss/{name}"].append(loss.detach())

                # Add sparsity regularizer
                if args.lasso:
                    loss += (
                        args.lasso
                        * th.cat([p.flatten() for p in lens[i].parameters()])
                        .abs()
                        .sum()
                    )

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

                log_dict = {k: th.stack(v).mean() for k, v in metrics.items()}

                # Log statistics about optimizer & probes
                for i, l in enumerate(lens):
                    name = "input" if i == 0 else f"{i - 1}.ffn"
                    states = [opt.state[p] for p in l.parameters()]

                    # Approximate the true gradient norm using SGD's moving average
                    log_dict["grad_norm/" + name] = th.cat(
                        [
                            # Undo PyTorch's scaling of the gradient by 1 / (1 - β). We
                            # also divide by the bias correction term (1 - β ** t).
                            (1 - β) * s["momentum_buffer"].flatten() / (1 - β**step)
                            for s in states
                        ]
                    ).norm()

                    assert isinstance(l, th.nn.Linear)
                    log_dict["bias_norm/" + name] = l.bias.data.norm()
                    log_dict["weight_norm/" + name] = l.weight.data.norm()

                metrics.clear()
                wandb.log(log_dict)

        # Make the problem strictly convex with projected gradient descent,
        # centering the affine transform and normalizing the scale
        lens.normalize_()

    if local_rank == 0:
        print(f"Saving lens to {args.output}")
        lens.save(args.output)
        th.save(opt.state_dict(), args.output / "optimizer.pt")


if __name__ == "__main__":
    parser = get_lens_parser()

    # Only print on rank 0
    with nullcontext() if local_rank == 0 else redirect_stdout(None):
        main(parser.parse_args())
