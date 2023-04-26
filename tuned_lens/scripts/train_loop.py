"""Training loop for training a TunedLens model against a transformer on a dataset."""
from collections import defaultdict
import enum
from typing import Optional
from pathlib import Path
from itertools import islice

from simple_parsing import field
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel
from tuned_lens import TunedLens
from tuned_lens.utils import shift_labels, shift_preds, maybe_all_reduce
from tuned_lens.scripts.ingredients import (
    Model,
    Data,
    Distributed,
    Optimizer,
)
import torch as th
from dataclasses import dataclass


class LossChoice(enum.Enum):
    """Options of what loss to select when training the model."""

    CE = "ce"
    KL = "kl"


@dataclass
class Train:
    """Training loop for the tuned lens."""

    model: Model
    """Model configuration."""

    data: Data
    """Data configuration."""

    opt: Optimizer
    """Optimizer configuration."""

    dist: Distributed
    """Configuration for how to distribute the training."""

    seed: int = 42
    """Random seed for data shuffling."""

    lens_name_or_path: Optional[str] = field(alias=["-l"], default=None)

    constant: Optional[bool] = field(action="store_true")
    """Train only the bias term."""

    num_steps: int = 250
    """Number of training steps."""

    output: Optional[Path] = field(alias=["-o"], default=None)
    """File to save the lenses to. Defaults to the model name."""

    pre_ln: Optional[bool] = field(action="store_true")
    """Apply layer norm before, and not after, each probe."""

    separate_unembeddings: Optional[bool] = field(action="store_true")
    """Learn a separate unembedding for each layer."""

    tokens_per_step: int = 2**18
    """Number of tokens per step."""

    wandb: Optional[str] = None
    """Name of run in Weights & Biases."""

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
            lens = TunedLens.from_model(model)
        else:
            lens = TunedLens.from_model_and_pretrained(model, self.lens_name_or_path)

        lens_size = sum(p.numel() * p.element_size() for p in lens.parameters())
        print(f"Tuned lens memory usage: {lens_size / 2 ** 20:.2f} MB per GPU")

        if self.constant:
            for probe in lens:
                probe.weight.requires_grad_(False)

        return lens

    def _init_logging(self, model_name: str, lens: TunedLens):
        """Initialize logging to weights and biases."""
        if not self.dist.primary or not self.wandb:
            return

        import wandb

        wandb.init(
            config=vars(self),
            group=model_name,
            name=self.wandb,
        )
        wandb.watch(lens)

    def _log(
        self,
        opt: th.optim.Optimizer,
        step: int,
        losses: dict[str, list[float]],
        tuned_lens: TunedLens,
        nats_to_bpb: float,
    ):
        """Log statistics about the training process to weights and biases."""
        if not self.dist.primary or not self.wandb:
            return

        import wandb

        log_dict = {}
        log_dict.update(
            {f"loss/{k}": th.tensor(v).mean() * nats_to_bpb for k, v in losses.items()}
        )

        # Log statistics about optimizer & probes
        for i, probe in enumerate(tuned_lens):
            name = "input" if i == 0 else f"{i - 1}.ffn"
            states = [opt.state[p] for p in probe.parameters()]

            # Approximate the true grad norm using the optimizer's moving
            # avg
            corr = 1 - self.opt.momentum**step
            if self.opt.optimizer == "sgd" and not self.opt.zero:
                log_dict["grad_norm/" + name] = th.cat(
                    [
                        # Undo PyTorch's scaling of the gradient by
                        # 1 / (1 - β)
                        (1 - self.opt.momentum) * s["momentum_buffer"].flatten() / corr
                        for s in states
                    ]
                ).norm()
            elif self.opt.optimizer == "adam" and not self.opt.zero:
                log_dict["grad_norm/" + name] = th.cat(
                    [s["exp_avg"].flatten() / corr for s in states]
                ).norm()

            if isinstance(probe, th.nn.Linear):
                log_dict["bias_norm/" + name] = probe.bias.data.norm()
                log_dict["weight_norm/" + name] = probe.weight.data.norm()

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

        global_batch_size = self.per_gpu_batch_size * self.dist.world_size
        grad_acc_steps, rem = divmod(samples_per_step, global_batch_size)
        if rem:
            # If the number of samples per step isn't divisible by the global batch
            # size, use ceil division and let the user know about it.
            grad_acc_steps += 1
            adjusted_count = grad_acc_steps * global_batch_size * tokens_per_sample
            print(
                f"Note: Increasing grad acc steps from {grad_acc_steps - 1} to "
                f"{grad_acc_steps} to maintain load balance across "
                f"{self.dist.world_size} GPUs."
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
        # Load model, tokenizer, data, and lens
        self.dist.init()
        model = tokenizer = data = lens = nats_to_bpb = model_name = None
        if self.dist.primary:
            # Let the primary processes populate the cache
            model, tokenizer = self.model.load()
            data, nats_to_bpb = self.data.load(tokenizer)
            lens = self.get_lens(model)

            *_, model_name = model.config.name_or_path.split("/")
            self._init_logging(model_name, lens)

        self.dist.barrier()  # Wait for primary to finish filling the cache

        if not self.dist.primary:
            # Let the non-primary processes load from the cache
            model, tokenizer = self.model.load(must_use_cache=True)
            data, nats_to_bpb = self.data.load(tokenizer)
            lens = self.get_lens(model)

        assert model and tokenizer and data and lens and nats_to_bpb

        # Shard the model using fully shared data parallel
        model = self.dist.shard_model(model)
        # Distribute the lens across the GPUS using distributed data parallel
        ddp_lens = self.dist.distribute_lens(lens)
        # Shard the dataset for use with distributed data parallel
        data = self.dist.shard_dataset(data)

        dl = DataLoader(
            data.shuffle(seed=self.seed),  # type: ignore[arg-type]
            batch_size=self.per_gpu_batch_size,
        )

        # Don't train the unembedding matrix or final layer norm
        params = [p for p in ddp_lens.parameters() if p.requires_grad]

        # Create the optimizer and scheduler
        opt = self.opt.create_optim(params)
        scheduler = self.opt.create_scheduler(opt, self.num_steps)

        tokens_per_sample = len(data[0]["input_ids"])

        grad_acc_steps = self.calculate_gradient_accumulation_steps(tokens_per_sample)

        if self.dist.cpu_offload and grad_acc_steps > 1:
            raise ValueError("CPU offloading cannot be used with gradient accumulation")

        losses = defaultdict(list)
        total_batches = self.num_steps * grad_acc_steps

        # Wait for all processes to finish setup
        self.dist.barrier()
        print("All processes have completed setup. Starting training.")

        # Main training loop
        pbar = tqdm(islice(dl, total_batches), desc="Training", total=total_batches)
        for batch_idx, batch in enumerate(pbar, start=1):
            assert isinstance(batch, dict)
            batch = self.dist.send_to_device(batch)
            with th.no_grad():
                output = model(**batch, output_hidden_states=True)

            final_logits = output.logits
            hidden_stats = output.hidden_states[:-1]

            shift = self.token_shift
            if self.loss == LossChoice.CE:
                labels = batch["input_ids"]

                # Predict the *next* token by default w/ cross entropy
                if shift is None:
                    shift = 1
            elif self.loss == LossChoice.KL:
                labels = final_logits.log_softmax(dim=-1)

                # Match the *current* token distribution by default
                if shift is None:
                    shift = 0
            else:
                raise NotImplementedError(f"Unknown loss {self.loss}")

            labels = shift_labels(labels, shift)

            # We do this sequentially to save VRAM
            for i, h in enumerate(hidden_stats):
                # bfloat16 has larger dynamic range than float16 and seems to be better
                # for computing log softmax & KL loss
                with th.autocast("cuda", dtype=th.bfloat16):
                    logits = shift_preds(ddp_lens(h, idx=i), shift)

                    if self.loss == LossChoice.CE:
                        loss = th.nn.functional.cross_entropy(
                            logits.flatten(0, -2), labels.flatten()
                        )
                    elif self.loss == LossChoice.KL:
                        loss = th.sum(
                            labels.exp() * (labels - logits.log_softmax(-1)), dim=-1
                        ).mean()
                    else:
                        raise NotImplementedError

                    # Log the loss *before* LASSO regularization
                    logging_loss = loss.detach()
                    logging_loss = maybe_all_reduce(logging_loss).item()
                    if self.dist.primary:
                        losses[f"translator_{i}"].append(logging_loss)

                    scaled_loss = loss / grad_acc_steps

                scaled_loss.backward()

            step, rem = divmod(batch_idx, grad_acc_steps)
            if rem == 0:
                th.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=False)
                scheduler.step()
                self._log(opt, step, losses, lens, nats_to_bpb)
                losses.clear()

            # Make the problem strictly convex with projected gradient descent,
            # centering the affine transform at each step

            # TODO this should be reviewed when we add support for other types of
            # normalization beyond layer norm
            lens.normalize_()

        if self.dist.primary:
            assert model_name is not None
            output = model_name if self.output is None else self.output
            print(f"Saving lens to {output}")
            lens.save(output)
