"""
ComfyUI SAM3 Custom Nodes - Official API Implementation
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

try:
    import folder_paths
except ImportError:
    class folder_paths:
        models_dir = "models"

from .sam3_utils import (
    SAM3ImageSegmenter,
    DepthEstimator,
    convert_to_segs
)


class SAM3ModelLoader:
    """Load SAM3 model with auto-download"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3"
    DESCRIPTION = "Load SAM3 model. Auto-downloads if not present. Requires HF authentication."

    def load_model(self, device: str) -> Tuple[Dict[str, Any]]:
        """Load SAM3 model"""
        try:
            # Load segmenter
            segmenter = SAM3ImageSegmenter(device=device)

            model_dict = {
                "segmenter": segmenter,
                "device": device
            }

            return (model_dict,)

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
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
            },
            "optional": {
                "text_prompt": ("STRING", {
                    "default": "object",
                    "multiline": False
                }),
                "point_mask": ("MASK",),
                "num_points": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1
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
        sam3_model: Dict[str, Any],
        image: torch.Tensor,
        mode: str,
        threshold: float = 0.5,
        text_prompt: str = "object",
        point_mask: Optional[torch.Tensor] = None,
        num_points: int = 5
    ) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """Perform segmentation"""
        try:
            segmenter = sam3_model["segmenter"]

            # Handle batch - process all images
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
                img = images[batch_idx]
                h, w, c = img.shape

                # Get corresponding mask if provided
                current_mask = None
                if point_mask is not None:
                    if len(point_mask.shape) == 3 and batch_idx < point_mask.shape[0]:
                        current_mask = point_mask[batch_idx]
                    elif len(point_mask.shape) == 2:
                        current_mask = point_mask

                # Segment based on mode
                if mode == "text":
                    if not text_prompt or text_prompt.strip() == "":
                        text_prompt = "object"

                    results = segmenter.segment_with_text(img, text_prompt, threshold)
                    label = f"text:{text_prompt}"

                elif mode == "points_from_mask":
                    if current_mask is None:
                        raise ValueError("point_mask required for points_from_mask mode")

                    # Extract points
                    points = segmenter.extract_points_from_mask(current_mask, num_points)

                    if len(points) == 0:
                        # Empty result for this frame
                        results = {"masks": [], "boxes": [], "scores": []}
                        label = "points:empty"
                    else:
                        results = segmenter.segment_with_points(img, points)
                        label = "points"

                else:  # auto
                    if text_prompt and text_prompt.strip() != "":
                        results = segmenter.segment_with_text(img, text_prompt, threshold)
                        label = f"auto:text:{text_prompt}"
                    elif current_mask is not None:
                        points = segmenter.extract_points_from_mask(current_mask, num_points)
                        if len(points) > 0:
                            results = segmenter.segment_with_points(img, points)
                            label = "auto:points"
                        else:
                            results = {"masks": [], "boxes": [], "scores": []}
                            label = "auto:empty"
                    else:
                        results = {"masks": [], "boxes": [], "scores": []}
                        label = "auto:empty"

                # Convert to SEGS
                segs = convert_to_segs(
                    results["masks"],
                    results["boxes"],
                    results["scores"],
                    (h, w),
                    label
                )
                all_segs.append(segs)

                # Create combined mask
                combined_mask = torch.zeros((h, w), dtype=torch.float32)
                for mask in results["masks"]:
                    mask_2d = self._ensure_2d_mask(mask, (h, w))
                    combined_mask = torch.maximum(combined_mask, mask_2d)

                all_combined_masks.append(combined_mask)

                # Create visualization
                vis_image = self._create_visualization(img, results["masks"])
                all_vis_images.append(vis_image.squeeze(0))

            # Stack results
            if batch_size == 1:
                final_segs = all_segs[0]
                final_vis = all_vis_images[0].unsqueeze(0)
                final_mask = all_combined_masks[0].unsqueeze(0)
            else:
                # For batches, return first SEGS (SEGS format doesn't support batches)
                # but stack visualizations and masks
                final_segs = all_segs[0]
                final_vis = torch.stack(all_vis_images, dim=0)
                final_mask = torch.stack(all_combined_masks, dim=0)

            return (final_segs, final_vis, final_mask)

        except Exception as e:
            error_msg = f"SAM3 Segmentation Error:\n{str(e)}"
            print(f"[SAM3] ERROR: {error_msg}")
            raise RuntimeError(error_msg)

    def _ensure_2d_mask(self, mask: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Ensure mask is 2D and correct size"""
        if isinstance(mask, torch.Tensor):
            mask_2d = mask.cpu()
        else:
            mask_2d = torch.from_numpy(np.array(mask))

        # Reduce dimensions
        while len(mask_2d.shape) > 2:
            mask_2d = mask_2d.squeeze(0) if mask_2d.shape[0] == 1 else mask_2d[0]

        # Resize if needed
        if mask_2d.shape != target_size:
            mask_2d = torch.nn.functional.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode="nearest"
            ).squeeze()

        return mask_2d.float()

    def _create_visualization(self, image: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        """Create colored visualization of masks"""
        img_np = image.cpu().numpy()
        h, w = img_np.shape[:2]

        overlay = np.zeros_like(img_np)

        for i, mask in enumerate(masks):
            mask_2d = self._ensure_2d_mask(mask, (h, w)).numpy()

            # Generate color
            color = np.array([
                ((i * 67) % 256) / 255.0,
                ((i * 113) % 256) / 255.0,
                ((i * 197) % 256) / 255.0
            ])

            mask_3d = np.stack([mask_2d] * 3, axis=-1)
            overlay += mask_3d * color

        # Blend
        result = img_np * 0.5 + overlay * 0.5
        result = np.clip(result, 0, 1)

        return torch.from_numpy(result).unsqueeze(0)


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
            # Initialize depth estimator
            if self.depth_estimator is None:
                self.depth_estimator = DepthEstimator()

            # Handle batch
            if len(image.shape) == 4:
                batch_size = image.shape[0]
                images = image
            else:
                batch_size = 1
                images = image.unsqueeze(0)

            all_depth_images = []
            all_depth_masks = []

            # Process each image in batch
            for batch_idx in range(batch_size):
                img = images[batch_idx]
                h, w, c = img.shape

                if mode == "full_image":
                    depth_map = self.depth_estimator.estimate_depth(img)

                else:  # per_segment
                    if segs is None:
                        raise ValueError("SEGS required for per_segment mode")

                    (img_w, img_h), segs_list = segs

                    if len(segs_list) == 0:
                        depth_map = self.depth_estimator.estimate_depth(img)
                    else:
                        depth_map = torch.zeros((h, w), dtype=torch.float32)

                        for seg in segs_list:
                            cropped_mask, crop_region, bbox, label, confidence = seg
                            x, y, crop_w, crop_h = crop_region

                            # Create full mask
                            full_mask = torch.zeros((h, w), dtype=torch.float32)

                            # Resize cropped mask
                            if cropped_mask.shape != (crop_h, crop_w):
                                resized_mask = torch.nn.functional.interpolate(
                                    cropped_mask.unsqueeze(0).unsqueeze(0),
                                    size=(crop_h, crop_w),
                                    mode="nearest"
                                ).squeeze()
                            else:
                                resized_mask = cropped_mask

                            # Place mask
                            end_y = min(y + crop_h, h)
                            end_x = min(x + crop_w, w)
                            actual_h = end_y - y
                            actual_w = end_x - x

                            full_mask[y:end_y, x:end_x] = resized_mask[:actual_h, :actual_w]

                            # Generate depth for segment
                            seg_depth = self.depth_estimator.estimate_depth(img, full_mask)
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

                # Also as mask [H, W]
                all_depth_masks.append(depth_map)

            # Stack results
            if batch_size == 1:
                final_depth_image = all_depth_images[0].unsqueeze(0)  # [1, H, W, C]
                final_depth_mask = all_depth_masks[0].unsqueeze(0)    # [1, H, W]
            else:
                final_depth_image = torch.stack(all_depth_images, dim=0)  # [B, H, W, C]
                final_depth_mask = torch.stack(all_depth_masks, dim=0)    # [B, H, W]

            return (final_depth_image, final_depth_mask)

        except Exception as e:
            error_msg = f"Depth Generation Error:\n{str(e)}"
            print(f"[SAM3] ERROR: {error_msg}")
            raise RuntimeError(error_msg)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3ModelLoader": SAM3ModelLoader,
    "SAM3Segmentation": SAM3Segmentation,
    "SAM3DepthMap": SAM3DepthMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3ModelLoader": "SAM3 Model Loader",
    "SAM3Segmentation": "SAM3 Segmentation",
    "SAM3DepthMap": "SAM3 Depth Map",
}
