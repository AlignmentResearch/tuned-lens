"""Training loop for training a TunedLens model against a transformer on a dataset."""
import dataclasses
import enum
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch as th
from simple_parsing import field
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchdata.dataloader2 import DataLoader2
from tqdm.auto import trange
from transformers import PreTrainedModel

import tuned_lens.scripts.ingredients as ing
from tuned_lens import TunedLens
from tuned_lens.utils import maybe_all_reduce, shift_labels, shift_preds


class LossChoice(enum.Enum):
    """Options of what loss to select when training the model."""

    CE = "ce"
    KL = "kl"


@dataclass
class State:
    """All of the stateful information in the training loop."""

    dataloader: DataLoader2
    lens: TunedLens
    opt: Optimizer
    scheduler: LambdaLR
    wandb_id: Optional[str]
    nats_to_bpb: float
    step: int = 0

    def load(self, snapshot_file: Path, device: th.device) -> None:
        """Load a snapshot file."""
        print(f"Loading snapshot from {snapshot_file}...")
        snapshot = th.load(snapshot_file, map_location=device)
        self.step = snapshot["step"]
        self.wandb_id = snapshot["wandb_id"]
        self.lens.load_state_dict(snapshot["lens"])
        self.opt.load_state_dict(snapshot["optim"])
        self.scheduler.load_state_dict(snapshot["scheduler"])
        self.dataloader.load_state_dict(snapshot["dataloader"])

    def save(self, snapshot_file: Path) -> None:
        """Save a snapshot file."""
        print(f"Saving snapshot to {snapshot_file}...")
        if isinstance(self.opt, ZeroRedundancyOptimizer):
            self.opt.consolidate_state_dict()

        th.save(
            {
                "lens": self.lens.state_dict(),
                "optim": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "dataloader": self.dataloader.state_dict(),
                "step": self.step,
                "wandb_id": self.wandb_id,
            },
            snapshot_file,
        )


@dataclass
class Train:
    """Training loop for the tuned lens."""

    model: ing.Model
    """Model configuration."""

    data: ing.Data
    """Data configuration."""

    opt: ing.Optimizer
    """Optimizer configuration."""

    dist: ing.Distributed
    """Configuration for how to distribute the training."""

    output: Path = field(alias=["-o"])
    """Directory to save the lenses to."""

    seed: int = 42
    """Random seed for data shuffling."""

    lens_name_or_path: Optional[str] = field(alias=["-l"], default=None)
    """Name of a pretrained lens to load for fine-tuning."""

    bias_only: Optional[bool] = field(action="store_true")
    """Train only the bias term."""

    num_steps: int = 250
    """Number of training steps."""

    tokens_per_step: int = 2**18
    """Number of tokens per step."""

    wandb: Optional[str] = None
    """Name of run in Weights & Biases."""

    token_shift: Optional[int] = None
    """How to shift the labels wrt the input tokens (1 = next token, 0 = current token,
    -1 = previous token, etc.)"""

    checkpoint_freq: Optional[int] = None
    """Steps between saving a checkpoint. If None, no checkpoints are saved."""

    checkpoint_dir: Optional[Path] = None
    """Directory to save checkpoints to. If None, will use <output>/checkpoints."""

    loss: LossChoice = LossChoice.KL
    """Loss function to use."""

    def __post_init__(self):
        """Set defaults for some fields."""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output / "checkpoints"

    def get_lens(self, model: PreTrainedModel) -> TunedLens:
        """Load or create a TunedLens model."""
        if self.lens_name_or_path is None:
            lens = TunedLens.from_model(model)
        else:
            lens = TunedLens.from_model_and_pretrained(model, self.lens_name_or_path)

        dtypes = {p.dtype for p in lens.parameters()}
        assert (
            len(dtypes) == 1
        ), f"Expected all parameters to have the same dtype, got {dtypes}"

        lens_dtype = next(iter(dtypes))
        lens_size = sum(p.numel() * p.element_size() for p in lens.parameters())

        # Include the optimizer state in the memory usage
        num_bytes = lens_size * (self.opt.per_parameter_optim_state_size() + 1)
        print(f"Tuned lens memory usage: {num_bytes / 2 ** 20:.2f} MB in {lens_dtype}")

        if self.bias_only:
            for probe in lens:
                probe.weight.requires_grad_(False)

        return lens

    def _get_wandb_id(self) -> Optional[str]:
        if not self.dist.primary or not self.wandb:
            return None

        from wandb.sdk.lib import runid

        return runid.generate_id()

    def _init_logging(self, model_name: str, lens: TunedLens, wandb_id: Optional[str]):
        """Initialize logging to weights and biases."""
        if not self.dist.primary or not self.wandb:
            return

        import wandb

        wandb.init(
            config=dataclasses.asdict(self),
            group=model_name,
            name=self.wandb,
            id=wandb_id,
            resume="allow",
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

    def snapshot(self, state: State):
        """Save a snapshot of the training process to disk."""
        if self.dist.primary:
            assert self.checkpoint_dir is not None
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state.save(self.checkpoint_dir / f"snapshot_{state.step}.pth")

    def load_recent_snapshot(self, state: State) -> None:
        """Load the most recent snapshot of the training process from disk."""
        assert self.checkpoint_dir is not None

        if not self.checkpoint_dir.exists():
            return None

        # Find the folder containing the most recent snapshot
        def sort_key_from_path(p: Path):
            if match := re.match(r".*snapshot_(\d+)\.pth", str(p)):
                return int(match.group(1))
            else:
                return -1

        snapshot_location = max(
            self.checkpoint_dir.glob("snapshot_*.pth"),
            key=sort_key_from_path,
            default=None,
        )

        if snapshot_location is None:
            return None

        state.load(snapshot_location, self.dist.device)

    def calculate_gradient_accumulation_steps(self, tokens_per_sample: int) -> int:
        """Calculate the number of batches of data to process before taking a step."""
        # chunk_and_tokenize ensures the samples are all the same length
        samples_per_step, rem = divmod(self.tokens_per_step, tokens_per_sample)
        if rem:
            raise ValueError(
                f"Number of tokens per step ({self.tokens_per_step:_}) must be "
                f"divisible by the number of tokens per sample ({tokens_per_sample})."
            )

        global_batch_size = self.dist.per_gpu_batch_size * self.dist.world_size
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

    def setup(self) -> tuple[State, Union[PreTrainedModel, FSDP], int]:
        """Initialize the training process."""
        self.dist.init()
        model = tokenizer = data = lens = nats_to_bpb = None

        # Annoyingly, FSDP is incompatible with the `device_map` parameter on
        # `from_pretrained`, because it adds forward hooks to the submodules that move
        # things around to different devices. But `bitsandbytes` requires `device_map`
        # to work at all. So we use `device_map` iff we're using FSDP.
        load_device = self.dist.device if not self.dist.fsdp else None

        if self.dist.primary:
            # Let the primary processes populate the cache
            model, tokenizer = self.model.load(load_device)
            data, nats_to_bpb = self.data.load(tokenizer)
            lens = self.get_lens(model)

        self.dist.barrier()  # Wait for primary to finish filling the cache

        if not self.dist.primary:
            # Let the non-primary processes load from the cache
            model, tokenizer = self.model.load(load_device, must_use_cache=True)
            data, nats_to_bpb = self.data.load(tokenizer)
            lens = self.get_lens(model)

        assert model and tokenizer and data and lens and nats_to_bpb

        dl = self.dist.data_loader(data)
        dl.seed(self.seed)
        params = [p for p in lens.parameters() if p.requires_grad]
        opt = self.opt.create_optim(params)
        scheduler = self.opt.create_scheduler(opt, self.num_steps)

        # Distribute the lens across the GPUS using distributed data parallel
        ddp_lens = self.dist.distribute_lens(lens)

        state = State(
            step=0,
            wandb_id=self._get_wandb_id(),
            lens=ddp_lens,  # type: ignore
            opt=opt,
            scheduler=scheduler,
            dataloader=dl,
            nats_to_bpb=nats_to_bpb,
        )

        self.load_recent_snapshot(state)

        # Shard the model using fully shared data parallel
        model = self.dist.shard_model(model)

        self._init_logging(
            model_name=self.model.name, lens=state.lens, wandb_id=state.wandb_id
        )

        tokens_per_sample = len(data[0]["input_ids"])
        grad_acc_steps = self.calculate_gradient_accumulation_steps(tokens_per_sample)
        self.dist.barrier()  # Wait for all processes to finish setup
        print("All processes have completed setup.")
        return state, model, grad_acc_steps

    def execute(self):
        """Trains a TunedLens model against a transformer on a dataset."""
        # Load model, tokenizer, data, and lens
        state, model, grad_acc_steps = self.setup()

        losses = defaultdict(list)
        init_batches = state.step * grad_acc_steps
        total_batches = self.num_steps * grad_acc_steps

        # Wait for all processes to finish setup
        self.dist.barrier()
        print("All processes have completed setup. Starting training.")

        # Main training loop
        t = trange(
            init_batches,
            total_batches,
            desc="Training",
            initial=init_batches,
            total=total_batches,
        )
        # TODO this currently silently fails if the dataloader is exhausted
        for batch_idx, batch in zip(t, state.dataloader):
            assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"

            with th.no_grad():
                batch = self.dist.send_to_device(batch)
                output = model(**batch, output_hidden_states=True)

            final_logits = output.logits
            hidden_states = output.hidden_states[:-1]

            shift = self.token_shift
            if self.loss == LossChoice.CE:
                labels = batch["input_ids"]

                # Predict the *next* token by default w/ cross entropy
                if shift is None:
                    shift = 1
            elif self.loss == LossChoice.KL:
                labels = final_logits.float().log_softmax(dim=-1)

                # Match the *current* token distribution by default
                if shift is None:
                    shift = 0
            else:
                raise NotImplementedError(f"Unknown loss {self.loss}")

            labels = shift_labels(labels, shift)

            # We do this sequentially to save VRAM
            for i, h in enumerate(hidden_states):
                # We use bfloat16 because it has a larger dynamic range than float16
                # and it seems to remove the need for doing grad scaling, which is very
                # annoying to set up in the context of multiple backward passes.
                with th.autocast(self.dist.device.type, dtype=th.bfloat16):
                    logits = shift_preds(state.lens(h, idx=i), shift)

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
            if rem == grad_acc_steps - 1:
                th.nn.utils.clip_grad_norm_(state.lens.parameters(), 1.0)
                state.opt.step()
                state.opt.zero_grad(set_to_none=False)
                state.scheduler.step()

                # Unwrap the lens from DDP if needed
                lens = getattr(state.lens, "module", state.lens)
                self._log(state.opt, step, losses, lens, state.nats_to_bpb)
                losses.clear()
                state.step = step + 1
                if (
                    self.checkpoint_freq
                    and step % self.checkpoint_freq == self.checkpoint_freq - 1
                ):
                    self.snapshot(state)

        if self.dist.primary:
            print(f"Saving lens to {self.output}")

            # Unwrap the lens from DDP if needed
            lens = getattr(state.lens, "module", state.lens)
            lens.save(self.output)
