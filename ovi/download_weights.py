import dataclasses
import logging
import pathlib
import time
from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


@dataclasses.dataclass
class DownloadedWeights:
    backbone_snapshot: str
    audio_vae_snapshot: str


def timed_download(repo_id: str, local_dir: str, allow_patterns: list):
    """Download files from HF repo and log time + destination."""
    logging.info(f"Starting download from {repo_id} into {local_dir}")
    start_time = time.time()

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    elapsed = time.time() - start_time
    logging.info(
        f"✅ Finished downloading {repo_id} "
        f"in {elapsed:.2f} seconds. Files saved at: {local_dir}"
    )


def needs_weights():
    return {
        "hkchengrex/MMAudio": [
            "ext_weights/best_netG.pt",
            "ext_weights/v1-16.pth",
        ],
        "chetwinlow1/Ovi": [
            "model.safetensors"
        ]
    }


def load_weights(cache_dir: pathlib.Path):
    for k, v in needs_weights().items():
        timed_download(k, str(cache_dir.absolute()), v)
