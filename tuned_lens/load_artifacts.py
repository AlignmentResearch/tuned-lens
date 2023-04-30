"""Load lens artifacts from the hub or locally storage."""
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


def load_lens_artifacts(
    resource_id: str,
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    config_file: str = "config.json",
    ckpt_file: str = "params.pt",
    subfolder: str = "lens",
    cache_dir: Optional[str] = None,
) -> tuple[Path, Path]:
    """First checks for lens resource locally then tries to download it from the hub.

    Args:
        resource_id: The id of the lens resource.
        repo_id: The repository to download the lens from. Defaults to
            'AlignmentResearch/tuned-lens'. However, this default can be overridden by
            setting the TUNED_LENS_REPO_ID environment variable.
        repo_type: The type of repository to download the lens from. Defaults to
            'space'. However, this default can be overridden by setting the
            TUNED_LENS_REPO_TYPE environment variable.
        config_file: The name of the config file in the folder contain the lens.
        ckpt_file: The name of the checkpoint file in the folder contain the lens.
        revision: The revision of the lens to download.
        subfolder: The subfolder of the repository to download the lens from.
        cache_dir: The directory to cache the lens in.

    Returns:
        * The path to the config.json file
        * The path to the params.pt file

    Raises:
        ValueError: if the lens resource could not be found.
    """
    if repo_id is None:
        if os.environ.get("TUNED_LENS_REPO_ID"):
            repo_id = os.environ["TUNED_LENS_REPO_ID"]
        else:
            repo_id = "AlignmentResearch/tuned-lens"

    if repo_type is None:
        if os.environ.get("TUNED_LENS_REPO_TYPE"):
            repo_type = os.environ["TUNED_LENS_REPO_TYPE"]
        else:
            repo_type = "space"

    # Fist check if the resource id is a path to a folder that exists
    local_path = Path(resource_id)
    if (local_path / config_file).exists() and (local_path / ckpt_file).exists():
        return local_path / config_file, local_path / ckpt_file

    subfolder = "/".join((subfolder, resource_id))
    params_path = hf_hub_download(
        filename=ckpt_file,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        subfolder=subfolder,
        cache_dir=cache_dir,
    )

    config_path = hf_hub_download(
        filename=config_file,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        subfolder=subfolder,
        cache_dir=cache_dir,
    )

    if config_path is not None and params_path is not None:
        return Path(config_path), Path(params_path)

    raise ValueError("Could not find lens resource locally or on the hf hub.")
