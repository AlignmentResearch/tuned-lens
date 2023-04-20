"""Shared configuration for the scripts."""
from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
from hashlib import md5
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel,
    MixedPrecision,
)
from typing import Optional, List
from dataclasses import dataclass
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from tuned_lens.data import (
    chunk_and_tokenize,
    compute_nats_to_bpb_ratio,
)
from tuned_lens.model_surgery import get_transformer_layers
import pickle
import shutil
import torch as th
import torch.distributed as dist

from simple_parsing import field

from tuned_lens.nn.lenses import TunedLens


def local_rank() -> int:
    """Get the local rank of the current process."""
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    """Get the world size from torch.distributed."""
    return dist.get_world_size() if dist.is_initialized() else 1


@dataclass
class Data:
    """Configuration for the dataset."""

    name: List[str] = field(default_factory=lambda: ["the_pile", "all"], nargs="*")
    """Name of dataset to use. Can either be a local .jsonl file or a name
    suitable to be passed to the HuggingFace load_dataset function."""

    split: str = "validation"
    """Split of the dataset to use."""

    text_column: str = "text"
    """Column of the dataset containing text to run the model on."""

    revision: Optional[str] = None

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

        processed = chunk_and_tokenize(dataset, tokenizer, text_key=self.text_column)
        nats_to_bpb = compute_nats_to_bpb_ratio(dataset, processed)

        print(f"Using nats per token to bits per byte ratio: {nats_to_bpb}")

        assert isinstance(processed, Dataset)
        return processed, nats_to_bpb


@dataclass
class Model:
    """Configuration for the model and tokenizer."""

    name: str
    """Name of model to use in the Huggingface Hub."""

    no_cache: bool = field(action="store_true")
    """Don't permanently cache the model on disk."""

    random_model: bool = field(action="store_true")
    """Use a randomly initialized model instead of pretrained weights."""

    revision: str = "main"
    """Git revision to use for pretrained models."""

    slow_tokenizer: bool = field(action="store_true")
    """Use a slow tokenizer."""

    tokenizer: Optional[str] = None
    """Name of pretrained tokenizer to use from the Huggingface Hub. If None, will use
    AutoTokenizer.from_pretrained('<model name>')."""

    tokenizer_type: Optional[str] = None
    """Name of tokenizer class to use. If None, will use AutoTokenizer."""

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Load the tokenizer from huggingface hub."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer or self.name,
            revision=self.revision,
            use_fast=not self.slow_tokenizer,
            tokenizer_type=self.tokenizer_type,
        )

        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        return tokenizer

    def load(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load the model and tokenizer."""
        # Deterministically choose a temporary cache directory shared
        # by all ranks using MD5 hash of the command line arguments
        if self.no_cache:
            cache_dir = f"/tmp/{md5(pickle.dumps(self)).hexdigest()}"
        else:
            cache_dir = None

        if self.random_model:
            print(f"Sampling random weights for model '{self.name}'...")
            config = AutoConfig.from_pretrained(self.name, revision=self.revision)
            model = AutoModelForCausalLM.from_config(  # type: ignore
                config, torch_dtype=th.float16
            )
        else:
            print(f"Loading pretrained weights for '{self.name}'...")
            model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                self.name,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                revision=self.revision,
                torch_dtype="auto",
            )

        # Make sure all ranks have the model
        if cache_dir:
            if dist.is_initialized():
                dist.barrier()
            if local_rank() == 0:
                shutil.rmtree(cache_dir)

        assert isinstance(model, PreTrainedModel)
        model.eval()
        model.requires_grad_(False)

        return model, self.load_tokenizer()


@dataclass
class Sharding:
    """Configuration for Fully Sharded Data Parallelism."""

    fsdp: bool = field(action="store_true")
    """Run the model with Fully Sharded Data Parallelism."""

    cpu_offload: bool = field(action="store_true")
    """Use CPU offloading. Must be combined with fsdp"""

    def shard_model(
        self, model: PreTrainedModel
    ) -> FullyShardedDataParallel | PreTrainedModel:
        """Shard the model using Fully Sharded Data Parallelism."""
        th.cuda.set_device(local_rank())

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
                device_id=local_rank(),
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
            model.to(local_rank())
            return model

    def shard_lens(self, lens: TunedLens) -> DDP | TunedLens:
        """Shard a tuned lens using DistributedDataParallel."""
        if world_size() > 1:
            return DDP(lens, device_ids=[local_rank()], find_unused_parameters=True)
        else:
            return lens

    def shard_dataset(self, dataset: Dataset) -> Dataset:
        """Shard the dataset based on local rank."""
        if dist.is_initialized():
            dataset = dataset.shard(dist.get_world_size(), local_rank())

        return dataset
