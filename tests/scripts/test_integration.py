from tuned_lens.__main__ import main
from pathlib import Path


def test_eval_subcommand(
    text_dataset_path: Path, gpt2_random_model_local_path: Path, tmp_path: Path
):
    # Note we do not specify a lens here, so we are using the logit lens
    args = (
        f"eval --data.name {text_dataset_path}"
        f" --model.name {gpt2_random_model_local_path}"
        " --limit 20 --max_length 128"
        f" --output {tmp_path}"
    )
    args = args.split()
    main(args)


def test_train_subcommand(
    text_dataset_path: Path, gpt2_random_model_local_path: Path, tmp_path: Path
):
    args = (
        f"train --data.name {text_dataset_path}"
        f" --model.name {gpt2_random_model_local_path}"
        " --max_length 128"
        f" --output {tmp_path}"
    )
    args = args.split()
    main(args)
