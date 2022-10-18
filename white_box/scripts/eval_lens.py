"""Train a set of tuned lenses for a language model."""

from accelerate.utils import send_to_device
from argparse import ArgumentParser
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_dataset
from itertools import islice
from white_box.data import chunk_and_tokenize, silence_datasets_messages
from white_box.model_surgery import get_final_layer_norm
from white_box import record_residual_stream, TunedLens
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from tqdm import tqdm
import torch as th


@th.no_grad()
def main():
    parser = ArgumentParser(
        description="Evaluate a set of tuned lenses for a language model."
    )
    parser.add_argument(
        "model_name", type=str, help="Name of model to use in the Huggingface Hub."
    )
    parser.add_argument(
        "--lens", type=str, help="File containing the lenses to evaluate."
    )
    parser.add_argument("--stats", type=str, help="File containing residual stats.")
    parser.add_argument(
        "--batch-size", type=int, default=6, help="Per-GPU batch size for eval."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        required=True,
        help="Name of dataset to use. Can either be a local .jsonl file or a name "
        "suitable to be passed to the HuggingFace load_dataset function.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for eval."
    )
    parser.add_argument(
        "--num-batches", type=int, default=1000, help="Number of batches to evaluate."
    )
    parser.add_argument(
        "--slow-tokenizer", action="store_true", help="Use a Python tokenizer."
    )
    parser.add_argument(
        "--split", type=str, default="validation", help="Split of the dataset to use."
    )
    parser.add_argument(
        "--sublayers", action="store_true", help="Create tuned lenses for sublayers."
    )
    parser.add_argument(
        "--text-column", type=str, default="text", help="Column of the dataset to use."
    )
    parser.add_argument(
        "--token-shift",
        type=int,
        default=1,
        help="How to shift the labels wrt the input tokens (1 = next token, "
        "0 = current token, -1 = previous token, etc.)",
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto"
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=not args.slow_tokenizer
    )

    # Just for type checking
    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    silence_datasets_messages()

    if args.dataset[0].endswith(".jsonl"):
        dataset = Dataset.from_json(*args.dataset)
    else:
        dataset = load_dataset(*args.dataset, split=args.split)
        if not isinstance(dataset, (Dataset, DatasetDict)):
            raise ValueError("Only Dataset and DatasetDict instances are supported.")

    processed = chunk_and_tokenize(dataset, tokenizer, text_key=args.text_column)
    dl = DataLoader(processed, batch_size=args.batch_size)  # type: ignore[arg-type]
    lens = TunedLens.load(args.lens).to(args.device) if args.lens else None

    # Exponential moving average of the loss
    ema = 0.0
    beta = 0.9
    metrics = defaultdict(list)

    # Reverse cumsum of the residual means for mean ablation
    biases = []
    if args.stats:
        residual_means = th.load(args.stats, map_location=args.device)
        acc = th.zeros_like(residual_means.mean.layers[0])
        for mean in reversed(residual_means.mean):
            biases.append(acc.clone())
            acc += mean

    pbar = tqdm(dl, desc="Evaluating")
    use_autocast = model.dtype == th.float16
    if use_autocast:
        pbar.write("Using fp16 inference for the model.")

    for batch in islice(dl, args.num_batches):
        batch = send_to_device(batch, args.device)
        with (
            record_residual_stream(model, sublayers=args.sublayers) as stream,
            th.autocast("cuda", enabled=use_autocast),
        ):
            model(**batch)

        total_loss = 0.0
        if lens is not None:
            logit_iter = lens.map(stream.map(lambda x: x.float()).items())
        else:
            for s, b in zip(reversed(stream), biases):
                s.add_(b)

            E = model.get_output_embeddings()
            ln = get_final_layer_norm(model.base_model)  # type: ignore[attr-defined]
            assert isinstance(ln, th.nn.LayerNorm)
            logit_iter = stream.map(lambda x: E(ln(x))).items()

        # We do this sequentially to save VRAM
        for name, logits in logit_iter:
            labels = batch["input_ids"]
            if args.token_shift > 0:
                labels = labels[:, args.token_shift :]
                logits = logits[:, : -args.token_shift]
            elif args.token_shift < 0:
                labels = labels[:, : args.token_shift]
                logits = logits[:, -args.token_shift :]

            loss = th.nn.functional.cross_entropy(
                logits.flatten(0, -2), labels.flatten()
            )
            total_loss += loss
            metrics[name].append(loss)

        total_loss = total_loss / len(stream)

        # Update the exponential moving average of the loss
        ema = beta * ema + (1 - beta) * float(total_loss)

        # Bias correction
        pbar.set_postfix(loss=ema / (1 - beta ** (pbar.n + 1)))
        pbar.update()

    avg_metrics = {k: th.stack(v).mean().item() for k, v in metrics.items()}
    print(avg_metrics)


if __name__ == "__main__":
    main()
