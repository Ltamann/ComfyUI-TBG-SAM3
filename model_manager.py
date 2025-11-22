"""
SAM3 Model Management Utilities
Handles local model detection, downloading, and path management
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import folder_paths
from huggingface_hub import hf_hub_download
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
    Get list of available SAM3 model checkpoints (files directly under models/sam3)
    """
    sam3_path = get_sam3_models_path()
    extensions = [".pt", ".pth", ".safetensors", ".bin"]

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


def download_sam3_model(hf_repo: str = "facebook/sam3") -> str:
    """
    Download only the main SAM3 checkpoint (sam3.pt) into:
        <models_dir>/sam3/sam3.pt

    Returns:
        The directory path containing sam3.pt (i.e. <models_dir>/sam3)
    """
    sam3_dir = get_sam3_models_path()  # e.g. .../models/sam3
    os.makedirs(sam3_dir, exist_ok=True)

    local_ckpt_path = os.path.join(sam3_dir, "sam3.pt")
    if os.path.isfile(local_ckpt_path):
        print(f"[SAM3] Using existing checkpoint: {local_ckpt_path}")
        return sam3_dir

    print(f"[SAM3] Downloading sam3.pt from {hf_repo} to {local_ckpt_path} ...")

    downloaded_path = hf_hub_download(
        repo_id=hf_repo,
        filename="sam3.pt",
        revision="main",
        local_dir=sam3_dir,
        local_dir_use_symlinks=False,
    )

    if downloaded_path != local_ckpt_path:
        import shutil
        shutil.move(downloaded_path, local_ckpt_path)

    print(f"[SAM3] Model downloaded successfully to: {local_ckpt_path}")
    return sam3_dir


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
