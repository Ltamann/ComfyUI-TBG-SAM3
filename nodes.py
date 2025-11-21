"""
ComfyUI SAM3 Custom Nodes - Official API Implementation
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from .model_manager import get_available_models, get_model_path, download_sam3_model

try:
    import folder_paths
except ImportError:
    class folder_paths:
        models_dir = "models"

from .sam3_utils import (
    SAM3ImageSegmenter,
    DepthEstimator,
    convert_to_segs,
    tensor_to_pil,
    pil_to_tensor,
    mask_to_tensor
)

import os

# Get base models path
try:
    import folder_paths
    base_models_folder = folder_paths.models_dir
except ImportError:
    base_models_folder = "models"


# Example: 20 visually distinct strong colors
COLORS = np.array([
    [1.00, 0.00, 0.00],  # Red
    [0.00, 1.00, 0.00],  # Green
    [0.00, 0.00, 1.00],  # Blue
    [1.00, 1.00, 0.00],  # Yellow
    [1.00, 0.00, 1.00],  # Magenta
    [0.00, 1.00, 1.00],  # Cyan
    [1.00, 0.65, 0.00],  # Orange
    [0.60, 0.33, 0.79],  # Purple
    [0.50, 0.50, 0.50],  # Gray
    [0.00, 0.00, 0.00],  # Black (avoid for overlays)
    [0.80, 0.36, 0.36],  # Light Red
    [0.35, 0.80, 0.36],  # Light Green
    [0.36, 0.36, 0.80],  # Light Blue
    [0.90, 0.70, 0.10],  # Pale Yellow
    [0.90, 0.10, 0.90],  # Pink
    [0.10, 0.90, 0.90],  # Light Cyan
    [1.00, 0.82, 0.57],  # Light Orange
    [0.68, 0.52, 0.98],  # Lavender
    [0.69, 0.69, 0.69],  # Silver
    [0.29, 0.19, 0.18],  # Brown
])

def get_palette_color(i):
    return COLORS[i % len(COLORS)]


class SAM3ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = []
        models.insert(0, "auto (API)")
        models.insert(1, "local (auto-download if missing)")
        return {
            "required": {
                "model_source": (models, {"default": "auto (API)"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3"
    DESCRIPTION = "Load SAM3 model locally or via API auto-download."

    def load_model(self, model_source: str, device: str):
        try:
            if model_source == "auto (API)":
                model_path = None
            elif model_source == "local (auto-download if missing)":
                model_path = os.path.join(base_models_folder, "sam3", "sam3_model", "model.safetensors")

                if not os.path.isfile(model_path):
                    print("[SAM3] Local model not found, downloading automatically...")
                    local_dir = download_sam3_model("facebook/sam3")
                    candidate_path = os.path.join(local_dir, "model.safetensors")
                    if os.path.isfile(candidate_path):
                        model_path = candidate_path
                    else:
                        raise RuntimeError(f"Expected SAM3 model checkpoint not found: {candidate_path}")
            else:
                model_path = get_model_path(model_source)
                if not model_path or not os.path.isfile(model_path):
                    raise RuntimeError(f"Local model file not found: {model_source}")

            segmenter = SAM3ImageSegmenter(device=device, model_path=model_path)
            print("[SAM3] ✓ Model loaded successfully")
            return (segmenter,)

        except Exception as e:
            error_msg = f"SAM3 Model Loading Error:\n{str(e)}"
            print(f"[SAM3] ERROR: {error_msg}")
            raise RuntimeError(error_msg)


class SAM3Segmentation:
    """SAM3 Segmentation with Impact Pack SEGS output"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "image": ("IMAGE",),
                "mode": (["text", "points_from_mask", "auto"], {"default": "text"}),
            },
            "optional": {
                "text_prompt": ("STRING", {
                    "default": "object",
                    "multiline": False
                }),
                "point_mask": ("MASK",),  # Optional mask input [B,H,W]
                "num_points": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("SEGS", "IMAGE", "MASK")
    RETURN_NAMES = ("segs", "visualization", "combined_mask")
    FUNCTION = "segment"
    CATEGORY = "SAM3"
    DESCRIPTION = "SAM3 segmentation with text or point prompts. Output: Impact Pack SEGS format."

    def segment(
            self,
            sam3_model,
            image: torch.Tensor,
            mode: str,
            text_prompt: str = "object",
            point_mask: Optional[torch.Tensor] = None,
            num_points: int = 5,
            threshold: float = 0.5
    ) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """Perform segmentation"""
        try:
            segmenter = sam3_model

            # Get device from model (respects user's choice in loader)
            model_device = segmenter.get_device()
            device = torch.device(model_device)

            # Move input image to model's device
            image = image.to(device)

            # Handle batch
            if len(image.shape) == 4:
                batch_size = image.shape[0]
                images = image
            else:
                batch_size = 1
                images = image.unsqueeze(0)

            all_vis_images = []
            all_combined_masks = []
            all_segs = []

            # Process each image in batch
            for batch_idx in range(batch_size):
                img = images[batch_idx]  # [H, W, C]
                h, w, c = img.shape

                # Get corresponding mask if provided and move to model device
                current_mask = None
                if point_mask is not None:
                    point_mask = point_mask.to(device)
                    if len(point_mask.shape) == 3 and batch_idx < point_mask.shape[0]:
                        current_mask = point_mask[batch_idx]
                    elif len(point_mask.shape) == 2:
                        current_mask = point_mask
                    else:
                        print(f"[SAM3] Warning: Unexpected mask shape: {point_mask.shape}")

                # Segment based on mode
                masks, boxes, scores = [], [], []

                if mode == "text":
                    if not text_prompt or text_prompt.strip() == "":
                        text_prompt = "object"

                    try:
                        masks, boxes, scores = segmenter.segment_image(img, text_prompt)
                        label = f"text:{text_prompt}"
                    except Exception as e:
                        print(f"[SAM3] Segmentation failed: {e}")
                        masks, boxes, scores = [], [], []
                        label = f"text:{text_prompt}:error"

                elif mode == "points_from_mask":
                    if current_mask is None:
                        print("[SAM3] Warning: points_from_mask mode requires point_mask input")
                        label = "points:no_mask"
                    else:
                        try:
                            points = segmenter.extract_points_from_mask(current_mask, num_points)
                            if len(points) == 0:
                                label = "points:empty"
                            else:
                                masks, boxes, scores = segmenter.segment_with_points(img, points)
                                label = f"points:{len(points)}"
                        except Exception as e:
                            print(f"[SAM3] Point segmentation failed: {e}")
                            label = "points:error"

                else:  # auto mode
                    if text_prompt and text_prompt.strip() != "":
                        try:
                            masks, boxes, scores = segmenter.segment_image(img, text_prompt)
                            label = f"auto:text:{text_prompt}"
                        except Exception as e:
                            print(f"[SAM3] Auto text segmentation failed: {e}")
                            label = "auto:text:error"
                    elif current_mask is not None:
                        try:
                            points = segmenter.extract_points_from_mask(current_mask, num_points)
                            if len(points) > 0:
                                masks, boxes, scores = segmenter.segment_with_points(img, points)
                                label = f"auto:points:{len(points)}"
                            else:
                                label = "auto:empty"
                        except Exception as e:
                            print(f"[SAM3] Auto point segmentation failed: {e}")
                            label = "auto:points:error"
                    else:
                        print("[SAM3] Warning: auto mode needs either text_prompt or point_mask")
                        label = "auto:no_input"

                # Ensure masks, boxes, scores are lists
                if not isinstance(masks, list):
                    masks = list(masks) if masks is not None else []
                if not isinstance(boxes, list):
                    boxes = list(boxes) if boxes is not None else []
                if not isinstance(scores, list):
                    scores = list(scores) if scores is not None else []

                # Filter by threshold
                if threshold > 0 and len(scores) > 0:
                    filtered = [(m, b, s) for m, b, s in zip(masks, boxes, scores) if s >= threshold]
                    if filtered:
                        masks, boxes, scores = zip(*filtered)
                        masks, boxes, scores = list(masks), list(boxes), list(scores)
                    else:
                        masks, boxes, scores = [], [], []
                        print(f"[SAM3] All results filtered out by threshold {threshold}")

                # Log if no results found
                if len(masks) == 0:
                    print(f"[SAM3] No objects found for prompt: '{text_prompt}' in mode: {mode}")

                # Convert to SEGS - handle empty results
                segs = convert_to_segs(masks, boxes, scores, (h, w), label)
                all_segs.append(segs)

                # Create combined mask - on model device
                combined_mask = torch.zeros((h, w), dtype=torch.float32, device=device)

                if len(masks) > 0:
                    for mask in masks:
                        mask_2d = self._ensure_2d_mask(mask, (h, w), device)
                        # Use additive masking with clamping to see all elements
                        combined_mask = combined_mask + mask_2d

                    # Clamp to [0, 1] range
                    combined_mask = torch.clamp(combined_mask, 0.0, 1.0)

                all_combined_masks.append(combined_mask)

                # Create visualization - on model device
                vis_image = self._create_visualization(img, masks, device)
                all_vis_images.append(vis_image.squeeze(0))

            # Stack results - all on model device
            if batch_size == 1:
                final_segs = all_segs[0]
                final_vis = all_vis_images[0].unsqueeze(0)
                final_mask = all_combined_masks[0].unsqueeze(0)
            else:
                final_segs = all_segs[0]
                final_vis = torch.stack(all_vis_images, dim=0)
                final_mask = torch.stack(all_combined_masks, dim=0)

            return (final_segs, final_vis, final_mask)

        except Exception as e:
            error_msg = f"SAM3 Segmentation Error:\n{str(e)}"
            print(f"[SAM3] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)

    def _ensure_2d_mask(self, mask, target_size: Tuple[int, int], device) -> torch.Tensor:
        """Ensure mask is 2D and correct size, on specified device"""
        mask_tensor = mask_to_tensor(mask)

        # Move to target device
        mask_tensor = mask_tensor.to(device)

        # Reduce dimensions
        while len(mask_tensor.shape) > 2:
            mask_tensor = mask_tensor.squeeze(0) if mask_tensor.shape[0] == 1 else mask_tensor[0]

        # Resize if needed
        if mask_tensor.shape != target_size:
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode="nearest"
            ).squeeze()

        return mask_tensor.float()

    def _create_visualization(self, image: torch.Tensor, masks: List, device) -> torch.Tensor:
        """Create colored visualization of masks, on specified device"""
        # Keep on device for processing
        img_np = image.cpu().numpy()
        h, w = img_np.shape[:2]

        # Start with black overlay
        overlay = np.zeros_like(img_np)

        # Track which pixels are masked
        any_mask = np.zeros((h, w), dtype=np.float32)

        if len(masks) > 0:
            for i, mask in enumerate(masks):
                # Pick color
                mask_2d = self._ensure_2d_mask(mask, (h, w), device).cpu().numpy()
                print(f"Visualizing mask {i} (max={mask_2d.max()}, unique={np.unique(mask_2d)})")

                # Update any_mask tracker
                any_mask = np.maximum(any_mask, mask_2d)

                color = get_palette_color(i)


                # Create 3-channel mask
                mask_3d = np.stack([mask_2d] * 3, axis=-1)

                # Add colored overlay (additive blending for overlapping regions)
                overlay += mask_3d * color

        # Clip overlay to valid range
        overlay = np.clip(overlay, 0, 1)

        # Blend with original image
        # Use stronger alpha for masked regions to make them more visible
        alpha = 0.6  # Increased from 0.5 for better visibility
        result = img_np * (1 - alpha) + overlay * alpha

        # Alternative: use any_mask to only show color on masked regions
        # any_mask_3d = np.stack([any_mask] * 3, axis=-1)
        # result = img_np * (1 - any_mask_3d * alpha) + overlay * alpha

        result = np.clip(result, 0, 1)

        # Convert back to tensor on model device
        result_tensor = torch.from_numpy(result.astype(np.float32)).to(device)
        return result_tensor.unsqueeze(0)


class SAM3DepthMap:
    """Generate depth maps for images or segments"""

    def __init__(self):
        self.depth_estimator = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["full_image", "per_segment"], {"default": "full_image"}),
                "normalize": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "segs": ("SEGS",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth_image", "depth_mask")
    FUNCTION = "generate_depth"
    CATEGORY = "SAM3"
    DESCRIPTION = "Generate depth maps. per_segment mode requires SEGS input."

    def generate_depth(
            self,
            image: torch.Tensor,
            mode: str,
            normalize: bool = True,
            segs: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate depth map"""
        try:
            if self.depth_estimator is None:
                self.depth_estimator = DepthEstimator()

            # Get device from input image
            device = image.device

            # Handle batch
            if len(image.shape) == 4:
                batch_size = image.shape[0]
                images = image
            else:
                batch_size = 1
                images = image.unsqueeze(0)

            all_depth_images = []
            all_depth_masks = []

            for batch_idx in range(batch_size):
                img = images[batch_idx]
                h, w, c = img.shape

                if mode == "full_image":
                    depth_map = self.depth_estimator.estimate_depth(img)
                    # Ensure on correct device
                    depth_map = depth_map.to(device)

                else:  # per_segment
                    if segs is None:
                        raise ValueError("SEGS required for per_segment mode")

                    (img_w, img_h), segs_list = segs

                    if len(segs_list) == 0:
                        depth_map = self.depth_estimator.estimate_depth(img)
                        depth_map = depth_map.to(device)
                    else:
                        depth_map = torch.zeros((h, w), dtype=torch.float32, device=device)

                        for seg in segs_list:
                            cropped_mask, crop_region, bbox, label, confidence = seg
                            x, y, crop_w, crop_h = crop_region

                            full_mask = torch.zeros((h, w), dtype=torch.float32, device=device)

                            if cropped_mask.shape != (crop_h, crop_w):
                                resized_mask = torch.nn.functional.interpolate(
                                    cropped_mask.unsqueeze(0).unsqueeze(0).to(device),
                                    size=(crop_h, crop_w),
                                    mode="nearest"
                                ).squeeze()
                            else:
                                resized_mask = cropped_mask.to(device)

                            end_y = min(y + crop_h, h)
                            end_x = min(x + crop_w, w)
                            actual_h = end_y - y
                            actual_w = end_x - x

                            full_mask[y:end_y, x:end_x] = resized_mask[:actual_h, :actual_w]

                            seg_depth = self.depth_estimator.estimate_depth(img, full_mask)
                            seg_depth = seg_depth.to(device)
                            depth_map = torch.maximum(depth_map, seg_depth)

                # Normalize
                if normalize:
                    depth_min = depth_map.min()
                    depth_max = depth_map.max()
                    if depth_max > depth_min:
                        depth_map = (depth_map - depth_min) / (depth_max - depth_min)

                # Convert to image [H, W, C]
                depth_image = depth_map.unsqueeze(-1).repeat(1, 1, 3)
                all_depth_images.append(depth_image)
                all_depth_masks.append(depth_map)

            # Stack results
            if batch_size == 1:
                final_depth_image = all_depth_images[0].unsqueeze(0)
                final_depth_mask = all_depth_masks[0].unsqueeze(0)
            else:
                final_depth_image = torch.stack(all_depth_images, dim=0)
                final_depth_mask = torch.stack(all_depth_masks, dim=0)

            return (final_depth_image, final_depth_mask)

        except Exception as e:
            error_msg = f"Depth Generation Error:\n{str(e)}"
            print(f"[SAM3] ERROR: {error_msg}")
            raise RuntimeError(error_msg)


class SAM3ModelDownloader:
    """Download SAM3 models from HuggingFace to local storage"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {
                    "default": "facebook/sam3",
                    "multiline": False
                }),
                "download": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "download_model"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True
    DESCRIPTION = """Download SAM3 model from HuggingFace to models/sam3/ folder.

Steps:
1. Request access: https://huggingface.co/facebook/sam3
2. Login: huggingface-cli login  
3. Set download=True to start download
4. After download, use SAM3ModelLoader to load the local model"""

    def download_model(self, repo_id: str, download: bool) -> Tuple[str]:
        """Download model to local storage"""
        from .model_manager import download_sam3_model, get_sam3_models_path

        if not download:
            return (f"Ready to download from: {repo_id}\nSet download=True to start",)

        try:
            print(f"[SAM3] Starting download: {repo_id}")
            local_path = download_sam3_model(repo_id)

            models_path = get_sam3_models_path()
            status = f"✓ Download successful!\n"
            status += f"Model saved to: {local_path}\n"
            status += f"Models folder: {models_path}\n"
            status += f"Restart ComfyUI and select model in SAM3ModelLoader"

            return (status,)

        except Exception as e:
            error_status = f"✗ Download failed:\n{str(e)}"
            print(f"[SAM3] {error_status}")
            return (error_status,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3ModelLoader": SAM3ModelLoader,
    "SAM3Segmentation": SAM3Segmentation,
    "SAM3DepthMap": SAM3DepthMap,
    "SAM3ModelDownloader": SAM3ModelDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3ModelLoader": "SAM3 Model Loader",
    "SAM3Segmentation": "SAM3 Segmentation",
    "SAM3DepthMap": "SAM3 Depth Map",
    "SAM3ModelDownloader": "SAM3 Model Downloader",
}
