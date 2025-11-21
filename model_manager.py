"""
SAM3 Model Management Utilities
Handles local model detection, downloading, and path management
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import folder_paths

# Define SAM3 model subfolder
SAM3_MODELS_DIR = "sam3"


def get_sam3_models_path() -> str:
    """Get the path to SAM3 models directory"""
    try:
        base_path = folder_paths.models_dir
    except:
        base_path = "models"

    sam3_path = os.path.join(base_path, SAM3_MODELS_DIR)

    # Create directory if it doesn't exist
    os.makedirs(sam3_path, exist_ok=True)

    return sam3_path


def get_available_models() -> List[str]:
    """
    Get list of available SAM3 model checkpoints
    Returns list of model filenames
    """
    sam3_path = get_sam3_models_path()

    # Supported model file extensions
    extensions = ['.pt', '.pth', '.safetensors', '.bin']

    models = ["auto (download from HuggingFace)"]

    if os.path.exists(sam3_path):
        for file in os.listdir(sam3_path):
            if any(file.endswith(ext) for ext in extensions):
                models.append(file)

    return models


def get_model_path(model_name: str) -> Optional[str]:
    """
    Get full path to model checkpoint
    Returns None for 'auto' (HuggingFace download)
    """
    if model_name == "auto (download from HuggingFace)" or model_name == "auto":
        return None

    sam3_path = get_sam3_models_path()
    model_path = os.path.join(sam3_path, model_name)

    if os.path.exists(model_path):
        return model_path

    return None


def download_sam3_model(
        model_name: str = "facebook/sam3",
        save_dir: Optional[str] = None
) -> str:
    """
    Download SAM3 model from HuggingFace to local directory

    Args:
        model_name: HuggingFace model repo (e.g., "facebook/sam3")
        save_dir: Directory to save model (defaults to models/sam3)

    Returns:
        Path to downloaded model
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub required for downloading models.\n"
            "Install: pip install huggingface_hub"
        )

    if save_dir is None:
        save_dir = get_sam3_models_path()

    print(f"[SAM3] Downloading {model_name} to {save_dir}...")
    print("[SAM3] This may take several minutes...")

    try:
        # Download entire model repository
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=save_dir,
            local_dir=os.path.join(save_dir, "sam3_model"),
            local_dir_use_symlinks=False
        )

        print(f"[SAM3] Model downloaded successfully to: {local_path}")
        return local_path

    except Exception as e:
        error_msg = f"Failed to download model: {str(e)}\n"
        error_msg += "\nMake sure you:\n"
        error_msg += "1. Have HuggingFace access token: huggingface-cli login\n"
        error_msg += "2. Requested access at: https://huggingface.co/facebook/sam3\n"
        raise RuntimeError(error_msg)


def get_model_info(model_name: str) -> dict:
    """Get information about a model"""
    model_path = get_model_path(model_name)

    info = {
        "name": model_name,
        "path": model_path,
        "exists": model_path is not None and os.path.exists(model_path),
        "size": None
    }

    if info["exists"]:
        info["size"] = os.path.getsize(model_path)

    return info
