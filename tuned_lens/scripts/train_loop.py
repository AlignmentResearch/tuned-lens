"""Training loop for training a TunedLens model against a transformer on a dataset."""
from collections import defaultdict
import enum
from typing import Optional
from pathlib import Path
from itertools import islice

from simple_parsing import field
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, PreTrainedModel
from tuned_lens import TunedLens
from tuned_lens.residual_stream import ResidualStream
from tuned_lens.utils import (
    maybe_all_reduce,
    shift_labels,
    shift_preds,
    send_to_device,
)
from tuned_lens.scripts.ingredients import (
    local_rank,
    world_size,
    Model,
    Data,
    Sharding,
)
import torch as th
from dataclasses import dataclass


class LossChoice(enum.Enum):
    """Options of what loss to select when training the model."""

    CE = "ce"
    KL = "kl"


class OptimizerOption(enum.Enum):
    """Options for the optimizer to use when training the model."""

    ADAM = "adam"
    SGD = "sgd"


@dataclass
class Train:
    """Training loop for the tuned lens."""

    model: Model
    """Model configuration."""

    data: Data
    """Data configuration."""

    sharding: Sharding
    """Sharding configuration."""

    seed: int = 42
    """Random seed for data shuffling."""

    lens_name_or_path: Optional[str] = field(alias=["-l"], default=None)

    constant: Optional[bool] = field(action="store_true")
    """Train only the bias term."""

    lasso: float = 0.0
    """LASSO (L1) regularization strength."""

    lr_scale: float = 1.0
    """The default LR (1e-3 for Adam, 1.0 for SGD) is scaled by this factor."""

    momentum: float = 0.9
    """Momentum coefficient for SGD, or beta1 for Adam."""

    num_steps: int = 250
    """Number of training steps."""

    optimizer: OptimizerOption = OptimizerOption.SGD
    """The type of optimizer to use."""

    output: Optional[Path] = field(alias=["-o"], default=None)
    """File to save the lenses to. Defaults to the model name."""

    pre_ln: Optional[bool] = field(action="store_true")
    """Apply layer norm before, and not after, each probe."""

    resume: Optional[Path] = None
    """File to resume training from."""

    separate_unembeddings: Optional[bool] = field(action="store_true")
    """Learn a separate unembedding for each layer."""

    tokens_per_step: int = 2**18
    """Number of tokens per step."""

    wandb: Optional[str] = None
    """Name of run in Weights & Biases."""

    warmup_steps: Optional[int] = None
    """Number of warmup steps. Defaults to min(0.1 * num_steps, 1000) for Adam and 0
    for SGD."""

    weight_decay: float = 1e-3
    """Weight decay coefficient."""

    zero: Optional[bool] = field(action="store_true")
    """Use ZeroRedundancyOptimizer."""

    token_shift: Optional[int] = None
    """How to shift the labels wrt the input tokens (1 = next token, 0 = current token,
    -1 = previous token, etc.)"""

    per_gpu_batch_size: int = 1
    """Number of samples to try to fit on a GPU at once."""

    loss: LossChoice = LossChoice.KL
    """Loss function to use."""

    def get_lens(self, model: PreTrainedModel) -> TunedLens:
        """Load or create a TunedLens model."""
        if self.lens_name_or_path is None:
            lens = TunedLens(model)
        else:
            lens = TunedLens.load(self.lens_name_or_path)

        lens_size = sum(p.numel() * p.element_size() for p in lens.parameters())
        print(f"Tuned lens memory usage: {lens_size / 2 ** 20:.2f} MB per GPU")

        if self.constant:
            for probe in lens:
                probe.weight.requires_grad_(False)

        return lens

    def _init_logging(self, model_name: str, lens: TunedLens):
        if local_rank() > 0 or not self.wandb:
            return

        import wandb

        wandb.init(
            config=vars(self),
            entity="eleutherai",
            group=model_name,
            name=self.wandb,
            project="tuned-lens",
        )
        wandb.watch(lens)

    def _log(self, opt, step, metrics, lens, nats_to_bpb):
        if local_rank() > 0 or not self.wandb:
            return

        import wandb

        log_dict = {k: th.stack(v).mean() * nats_to_bpb for k, v in metrics.items()}

        # Log statistics about optimizer & probes
        for i, probe in enumerate(lens):
            name = "input" if i == 0 else f"{i - 1}.ffn"
            states = [opt.state[p] for p in probe.parameters()]

            # Approximate the true grad norm using the optimizer's moving
            # avg
            corr = 1 - self.momentum**step
            if self.optimizer == "sgd" and not self.zero:
                log_dict["grad_norm/" + name] = th.cat(
                    [
                        # Undo PyTorch's scaling of the gradient by
                        # 1 / (1 - β)
                        (1 - self.momentum) * s["momentum_buffer"].flatten() / corr
                        for s in states
                    ]
                ).norm()
            elif self.optimizer == "adam" and not self.zero:
                log_dict["grad_norm/" + name] = th.cat(
                    [s["exp_avg"].flatten() / corr for s in states]
                ).norm()

            if isinstance(probe, th.nn.Linear):
                log_dict["bias_norm/" + name] = probe.bias.data.norm()
                log_dict["weight_norm/" + name] = probe.weight.data.norm()

        metrics.clear()
        wandb.log(log_dict)

    def calculate_gradient_accumulation_steps(self, tokens_per_sample: int) -> int:
        """Calculate the number of batches of data to process before taking a step."""
        # chunk_and_tokenize ensures the samples are all the same length
        samples_per_step, rem = divmod(self.tokens_per_step, tokens_per_sample)
        if rem:
            raise ValueError(
                f"Number of tokens per step ({self.tokens_per_step:_}) must be "
                f"divisible by the number of tokens per sample ({tokens_per_sample})."
            )

        global_batch_size = self.per_gpu_batch_size * world_size()
        grad_acc_steps, rem = divmod(samples_per_step, global_batch_size)
        if rem:
            # If the number of samples per step isn't divisible by the global batch
            # size, use ceil division and let the user know about it.
            grad_acc_steps += 1
            adjusted_count = grad_acc_steps * global_batch_size * tokens_per_sample
            print(
                f"Note: Increasing grad acc steps from {grad_acc_steps - 1} to "
                f"{grad_acc_steps} to maintain load balance across {world_size()} "
                "GPUs."
            )
            print(
                f"Using {adjusted_count:_} tokens per training step "
                f"({self.tokens_per_step:_} requested)."
            )
        else:
            print(f"Gradient accumulation steps: {grad_acc_steps}")
            print(f"Using {self.tokens_per_step:_} tokens per training step.")
        return grad_acc_steps

    def execute(self):
        """Trains a TunedLens model against a transformer on a dataset."""
        model, tokenizer = self.model.load()
        data, nats_to_bpb = self.data.load(tokenizer)
        lens = self.get_lens(model)

        dl = DataLoader(
            data.shuffle(seed=self.seed),  # type: ignore[arg-type]
            batch_size=self.per_gpu_batch_size,
        )

        *_, model_name = model.config.name_or_path.split("/")

        self._init_logging(model_name, lens)

        # Shard across GPUs
        model = self.sharding.shard_model(model)
        data = self.sharding.shard_dataset(data)
        ddp_lens = self.sharding.shard_lens(lens)

        # Don't train the unembedding matrix or final layer norm
        params = [p for p in ddp_lens.parameters() if p.requires_grad]

        β = self.momentum
        if self.optimizer == "sgd":
            config = dict(
                # PyTorch's implementation effectively scales the LR by 1 / (1 - β),
                # so we undo that here. See https://www.youtube.com/watch?v=k8fTYJPd3_I
                # for discussion. Once we do this, the optimal LR seems to be unity.
                lr=self.lr_scale * (1 - β),
                momentum=β,
                # Empirically Nesterov momentum seems to improve convergence speed.
                nesterov=True,
                weight_decay=self.weight_decay,
            )
            opt_class = th.optim.SGD
        elif self.optimizer == "adam":
            config = dict(
                # Helps convergence slightly by ensuring that the LR actually decays
                amsgrad=True,
                betas=(β, 0.999),
                lr=self.lr_scale * 1e-3,
                weight_decay=self.weight_decay,
            )
            opt_class = th.optim.Adam
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'")

        if self.zero:
            opt = ZeroRedundancyOptimizer(params, optimizer_class=opt_class, **config)
        else:
            opt = opt_class(params, **config)  # type: ignore[call-arg]

        if self.warmup_steps is None:
            # Adam generally performs poorly without an LR warmup
            if self.optimizer == "adam":
                self.warmup_steps = min(1000, self.num_steps // 5)
                print(f"Using {self.warmup_steps} LR warmup steps for Adam")
            else:
                self.warmup_steps = 0

        scheduler = get_linear_schedule_with_warmup(
            opt, self.warmup_steps, self.num_steps - self.warmup_steps
        )
        if self.resume:
            assert self.resume.is_dir()

            print(f"Loading checkpoint from {self.resume}")
            ddp_lens.load_state_dict(th.load(self.resume))

        tokens_per_sample = len(data[0]["input_ids"])

        grad_acc_steps = self.calculate_gradient_accumulation_steps(tokens_per_sample)

        if self.sharding.cpu_offload and grad_acc_steps > 1:
            raise ValueError("CPU offloading cannot be used with gradient accumulation")

        metrics = defaultdict(list)
        total_batches = self.num_steps * grad_acc_steps

        pbar = tqdm(islice(dl, total_batches), desc="Training", total=total_batches)
        for batch_idx, batch in enumerate(pbar, start=1):
            assert isinstance(batch, dict)
            batch = send_to_device(batch, th.device(local_rank()))
            with th.autocast("cuda"):
                output = model(**batch, output_hidden_states=True)

            final_logits = output.logits
            stream = ResidualStream(
                embeddings=output.hidden_states[0], layers=output.hidden_states[1:-1]
            )

            shift = self.token_shift
            if self.loss == "ce":
                labels = batch["input_ids"]

                # Predict the *next* token by default w/ cross entropy
                if shift is None:
                    shift = 1
            elif self.loss == "kl":
                labels = final_logits.log_softmax(dim=-1)

                # Match the *current* token distribution by default
                if shift is None:
                    shift = 0
            else:
                raise NotImplementedError(f"Unknown loss {self.loss}")

            labels = shift_labels(labels, shift)

            # We do this sequentially to save VRAM
            for i, (name, h) in enumerate(stream.items()):
                # bfloat16 has larger dynamic range than float16 and seems to be better
                # for computing log softmax & KL loss
                with th.autocast("cuda", dtype=th.bfloat16):
                    logits = shift_preds(ddp_lens(h, idx=i), shift)

                    if self.loss == "ce":
                        loss = th.nn.functional.cross_entropy(
                            logits.flatten(0, -2), labels.flatten()
                        )
                    elif self.loss == "kl":
                        loss = th.sum(
                            labels.exp() * (labels - logits.log_softmax(-1)), dim=-1
                        ).mean()
                    else:
                        raise NotImplementedError

                    # Log the loss *before* LASSO regularization
                    logging_loss = loss.detach()
                    maybe_all_reduce(logging_loss)
                    if local_rank() == 0:
                        metrics[f"loss/{name}"].append(logging_loss)

                    # Add sparsity regularizer
                    if self.lasso:
                        param_vec = th.nn.utils.parameters_to_vector(
                            lens[i].parameters()
                        )
                        loss += self.lasso * param_vec.abs().sum()

                    scaled_loss = loss / grad_acc_steps

                scaled_loss.backward()

            step, rem = divmod(batch_idx, grad_acc_steps)
            if rem == 0:
                th.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=False)
                scheduler.step()
                self._log(opt, step, metrics, lens, nats_to_bpb)

            # Make the problem strictly convex with projected gradient descent,
            # centering the affine transform at each step
            lens.normalize_()

        if local_rank() == 0:
            output = model_name if self.output is None else self.output
            print(f"Saving lens to {self.output}")
            lens.save(output)
