"""Shared configuration for the scripts."""
import os
from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel,
    MixedPrecision,
)
import enum
from typing import Optional, Union
from dataclasses import dataclass
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from tuned_lens.data import (
    chunk_and_tokenize,
)

from tuned_lens.utils import (
    send_to_device,
    TreeType,
)
from tuned_lens.model_surgery import get_transformer_layers
import torch as th
import torch.distributed as dist

from simple_parsing import field

from tuned_lens.nn.lenses import Lens


@dataclass
class Data:
    """Configuration for the dataset."""

    name: list[str] = field(default_factory=lambda: ["the_pile", "all"], nargs="*")
    """Name of dataset to use. Can either be a local .jsonl file or a name
    suitable to be passed to the HuggingFace load_dataset function."""

    split: str = "validation"
    """Split of the dataset to use."""

    text_column: str = "text"
    """Column of the dataset containing text to run the model on."""

    revision: Optional[str] = None
    """The revision of the dataset to use"""

    max_length: int = 2048
    """The maximum length of the input sequences."""

    def load(self, tokenizer: PreTrainedTokenizerBase) -> tuple[Dataset, float]:
        """Load the dataset, tokenize it and compute nats_to_bpb."""
        print(f"Loading dataset '{' '.join(self.name)}'")

        if len(self.name) == 1 and self.name[0].endswith(".jsonl"):
            dataset = Dataset.from_json(self.name[0])
            assert isinstance(dataset, Dataset)
        else:
            dataset = load_dataset(*self.name, split=self.split, revision=self.revision)
            if not isinstance(dataset, (Dataset, DatasetDict)):
                raise ValueError(
                    "Only Dataset and DatasetDict instances are supported."
                )

        processed, nats_to_bpb = chunk_and_tokenize(
            dataset, tokenizer, text_key=self.text_column, max_length=self.max_length
        )

        print(f"Using nats per token to bits per byte ratio: {nats_to_bpb}")

        assert isinstance(processed, Dataset)

        return processed, nats_to_bpb


@dataclass
class Model:
    """Configuration for the model and tokenizer."""

    name: str
    """Name of model to use in the Huggingface Hub."""

    revision: str = "main"
    """Git revision to use for pretrained models."""

    slow_tokenizer: bool = field(action="store_true")
    """Use a slow tokenizer."""

    tokenizer: Optional[str] = None
    """Name of pretrained tokenizer to use from the Huggingface Hub. If None, will use
    AutoTokenizer.from_pretrained('<model name>')."""

    tokenizer_type: Optional[str] = None
    """Name of tokenizer class to use. If None, will use AutoTokenizer."""

    def load_tokenizer(self, must_use_cache: bool = False) -> PreTrainedTokenizerBase:
        """Load the tokenizer from huggingface hub."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer or self.name,
            revision=self.revision,
            use_fast=not self.slow_tokenizer,
            tokenizer_type=self.tokenizer_type,
            local_files_only=must_use_cache,
        )

        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        return tokenizer

    def load(
        self, must_use_cache: bool = False
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load the model and tokenizer."""
        print(f"Loading pretrained weights for '{self.name}'...")
        model = AutoModelForCausalLM.from_pretrained(  # type: ignore
            self.name,
            low_cpu_mem_usage=True,
            revision=self.revision,
            torch_dtype="auto",
            local_files_only=must_use_cache,
        )

        assert isinstance(model, PreTrainedModel)
        model.eval()
        model.requires_grad_(False)

        return model, self.load_tokenizer(must_use_cache=must_use_cache)


class OptimizerOption(enum.Enum):
    """Options for the optimizer to use when training the model."""

    ADAM = "adam"
    SGD = "sgd"


@dataclass
class Optimizer:
    """Configuration for the optimizer."""

    weight_decay: float = 1e-3
    """Weight decay coefficient."""

    lr_scale: float = 1.0
    """The default LR (1e-3 for Adam, 1.0 for SGD) is scaled by this factor."""

    momentum: float = 0.9
    """Momentum coefficient for SGD, or beta1 for Adam."""

    zero: Optional[bool] = field(action="store_true")
    """Use ZeroRedundancyOptimizer."""

    optimizer: OptimizerOption = OptimizerOption.SGD
    """The type of optimizer to use."""

    warmup_steps: Optional[int] = None
    """Number of warmup steps. Defaults to min(0.2 * num_steps, 1000) for Adam and 0
    for SGD."""

    def create_scheduler(
        self, opt: th.optim.Optimizer, num_steps: int
    ) -> th.optim.lr_scheduler.LambdaLR:
        """Create the LR scheduler."""
        if self.warmup_steps is None:
            # Adam generally performs poorly without an LR warmup
            if self.optimizer == "adam":
                self.warmup_steps = min(1000, num_steps // 5)
                print(f"Using {self.warmup_steps} LR warmup steps for Adam")
            else:
                self.warmup_steps = 0

        scheduler = get_linear_schedule_with_warmup(
            opt, self.warmup_steps, num_steps - self.warmup_steps
        )

        return scheduler

    def create_optim(self, params: list[th.nn.Parameter]) -> th.optim.Optimizer:
        """Create the optimizer."""
        # Don't train things that don't need gradients
        β = self.momentum
        if self.optimizer == OptimizerOption.SGD:
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
        elif self.optimizer == OptimizerOption.ADAM:
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

        return opt


@dataclass
class Distributed:
    """Configuration and utilities for distributing the model."""

    fsdp: bool = field(action="store_true")
    """Run the model with Fully Sharded Data Parallelism."""

    cpu_offload: bool = field(action="store_true")
    """Use CPU offloading. Must be combined with fsdp"""

    @property
    def rank(self) -> int:
        """The rank of this process.

        Note that in general this is not the same as the local rank.
        However, for single-node training, the local rank is the same as the
        global rank.
        """
        return int(os.environ["RANK"]) if dist.is_initialized() else 0

    @property
    def local_rank(self) -> int:
        """The local rank of this process."""
        return int(os.environ["LOCAL_RANK"]) if dist.is_initialized() else 0

    @property
    def world_size(self) -> int:
        """Get the world size from torch.distributed."""
        return int(os.environ["WORLD_SIZE"]) if dist.is_initialized() else 1

    @property
    def primary(self) -> bool:
        """Whether this is the rank 0 process."""
        return self.rank == 0

    @property
    def device(self) -> th.device:
        """The device associated with this process."""
        return th.device("cuda")

    def shard_model(
        self, model: PreTrainedModel
    ) -> Union[FullyShardedDataParallel, PreTrainedModel]:
        """Shard the model using Fully Sharded Data Parallelism."""
        th.cuda.set_device(self.rank)

        if self.fsdp:
            _, layers = get_transformer_layers(model)
            layer_cls = type(layers[0])
            print(f"Using '{layer_cls.__name__}' for transformer_auto_wrap_policy.")
            return FullyShardedDataParallel(
                model,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
                ),
                cpu_offload=CPUOffload(offload_params=self.cpu_offload),
                device_id=self.rank,
                # This turns out to be important for training speed
                forward_prefetch=True,
                mixed_precision=MixedPrecision(
                    param_dtype=th.float16,
                    reduce_dtype=th.float16,
                    buffer_dtype=th.float16,
                ),
            )
        elif self.cpu_offload:
            raise ValueError("CPU offload requires FSDP.")
        else:
            model.to(self.rank)
            return model

    def distribute_lens(self, lens: Lens) -> Union[DDP, Lens]:
        """Distribute the lens using DistributedDataParallel and send lens to device."""
        if self.world_size > 1:
            lens.to(self.device)
            return DDP(lens, device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            return lens.to(self.device)

    def shard_dataset(self, dataset: Dataset) -> Dataset:
        """Shard the dataset based on local rank."""
        if dist.is_initialized():
            dataset = dataset.shard(self.world_size, self.rank)
        return dataset

    def init(self) -> None:
        """Initialize distributed process group if started with elastic launch."""
        # Support both distributed and non-distributed training
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            dist.init_process_group("nccl")
            th.cuda.set_device(self.local_rank)

    def barrier(self) -> None:
        """Barrier for all processes."""
        if dist.is_initialized():
            dist.barrier()

    def send_to_device(self, pytree: TreeType) -> TreeType:
        """Move pytree to the current device."""
        return send_to_device(pytree, self.device)
