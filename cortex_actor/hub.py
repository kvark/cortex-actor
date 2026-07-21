"""Load released Cortex checkpoints from the Hugging Face Hub."""
import torch

from .model import cortex_from_args


def from_hub(repo_id: str = "mad-bot/cortex", filename: str = "cortex_quake_bc.pt"):
    """Download a released checkpoint and return ``(model, checkpoint_dict)``.

    The model is reconstructed from the checkpoint's stored architecture
    arguments, loaded strictly, and returned in eval mode. The checkpoint dict
    keeps the action schema and training-alignment constants deployment needs.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "loading from the Hub requires huggingface_hub: pip install huggingface_hub"
        ) from e

    # config.json mirrors the checkpoint's architecture args and action schema;
    # it is also the Hub's download-counting query file.
    hf_hub_download(repo_id, "config.json")
    path = hf_hub_download(repo_id, filename)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = cortex_from_args(ck["args_dict"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
