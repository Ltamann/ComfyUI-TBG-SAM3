"""
SAM3 Utility Functions - Based on Official SAM3 API
Enhanced with proper error handling and tensor conversions
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image

try:
    import folder_paths
except ImportError:
    class folder_paths:
        models_dir = "models"


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor [H,W,C] range [0,1] to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor [H,W,C] range [0,1]"""
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np)


def mask_to_tensor(mask) -> torch.Tensor:
    """Convert various mask formats to torch tensor"""
    if isinstance(mask, torch.Tensor):
        return mask.float()
    elif isinstance(mask, np.ndarray):
        return torch.from_numpy(mask).float()
    elif isinstance(mask, Image.Image):
        return torch.from_numpy(np.array(mask)).float()
    else:
        return torch.tensor(mask).float()


class SAM3ImageSegmenter:
    """SAM3 Image Segmentation using official API"""

    def __init__(self, device: str = "cuda", model_path: Optional[str] = None):
        """Initialize SAM3 image segmenter"""
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model(model_path)

    def get_device(self) -> str:
        """Get the device this model is running on"""
        return self.device

    def _load_model(self, model_path: Optional[str] = None):
        """Load SAM3 model - supports both API and local loading"""
        try:
            # Import SAM3 modules
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except ImportError:
                from sam3 import build_sam3_image_model, Sam3Processor

            print(f"[SAM3] Loading SAM3 image model on {self.device}...")

            # API AUTO-DOWNLOAD MODE (model_path is None)
            if model_path is None:
                print("[SAM3] Auto-downloading from HuggingFace...")
                self.model = build_sam3_image_model(device=self.device)
                self.processor = Sam3Processor(self.model)
                print("[SAM3] ✓ Model loaded successfully (API mode)")
                return

            # LOCAL MODEL LOADING MODE
            print(f"[SAM3] Loading from local checkpoint: {model_path}")

            # Build empty model first
            self.model = build_sam3_image_model(device=self.device)

            # Load weights
            if model_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path, device=str(self.device))
                except ImportError:
                    print("[SAM3] WARNING: safetensors not installed, using torch.load")
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            else:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(state_dict, strict=False)
            self.processor = Sam3Processor(self.model)
            print("[SAM3] ✓ Model loaded successfully (local mode)")

        except ImportError as e:
            error_msg = f"SAM3 not installed: {e}\n"
            error_msg += "Install: pip install git+https://github.com/facebookresearch/sam3.git"
            raise RuntimeError(error_msg)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {str(e)}")

    def segment_image(self, image, text_prompt: str):
        if isinstance(image, torch.Tensor):
            pil_image = tensor_to_pil(image)
        else:
            pil_image = image

        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        print(f"[SAM3 DEBUG] segment_image output keys: {list(output.keys())}")
        print(f"[SAM3 DEBUG] Number of masks found: {len(output['masks'])}")
        print(f"[SAM3 DEBUG] Scores: {output.get('scores', 'N/A')}")

        return output["masks"], output["boxes"], output["scores"]

    def segment_with_points(self, image, points: List[Tuple[int, int]],
                           point_labels: Optional[List[int]] = None):
        """Segment using point prompts"""
        if point_labels is None:
            point_labels = [1] * len(points)

        if isinstance(image, torch.Tensor):
            pil_image = tensor_to_pil(image)
        else:
            pil_image = image

        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_point_prompt(
            state=inference_state,
            points=points,
            labels=point_labels
        )

        return output["masks"], output["boxes"], output["scores"]

    def extract_points_from_mask(self, mask: torch.Tensor, num_points: int = 5) -> List[Tuple[int, int]]:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        mask_np = mask_np.squeeze()
        y_coords, x_coords = np.where(mask_np > 0.5)
        total_points = len(y_coords)
        if total_points == 0:
            print("[SAM3] extract_points_from_mask: No foreground pixels found in mask!")
            return []

        if total_points <= num_points:
            indices = range(total_points)
        else:
            indices = np.linspace(0, total_points - 1, num_points, dtype=int)

        points = [(int(x_coords[i]), int(y_coords[i])) for i in indices]
        print(f"[SAM3] extract_points_from_mask: returning {len(points)} points.")
        return points


class DepthEstimator:
    """Depth map generation using MiDaS"""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load MiDaS depth model"""
        try:
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            model_name = "Intel/dpt-hybrid-midas"
            print(f"[SAM3] Loading depth model: {model_name}")
            self.processor = DPTImageProcessor.from_pretrained(model_name)
            self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("[SAM3] Depth model loaded")
        except Exception as e:
            print(f"[SAM3] Could not load depth model: {e}")
            self.model = None

    def estimate_depth(self, image, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Estimate depth map"""
        if self.model is None:
            # Return dummy depth if model not loaded
            if isinstance(image, torch.Tensor):
                h, w = image.shape[:2]
            else:
                h, w = np.array(image).shape[:2]
            return torch.zeros((h, w), dtype=torch.float32)

        # Convert to PIL
        if isinstance(image, torch.Tensor):
            pil_image = tensor_to_pil(image)
        else:
            pil_image = image

        # Process
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth

        # Resize to original size
        if isinstance(image, torch.Tensor):
            target_size = image.shape[:2]
        else:
            target_size = np.array(image).shape[:2]

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=target_size,
            mode="bicubic",
            align_corners=False
        ).squeeze()

        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) > 2:
                mask = mask[..., 0]
            depth = depth * mask.to(self.device)

        return depth.cpu()


def convert_to_segs(
    masks: List,
    boxes: List,
    scores: List[float],
    image_size: Tuple[int, int],
    label: str = "sam3"
) -> Tuple[Tuple[int, int], List[Tuple]]:
    """
    Convert SAM3 outputs to Impact Pack SEGS format

    SEGS format: ((width, height), [seg1, seg2, ...])
    Each seg: (cropped_mask, crop_region, bbox, label, confidence)
    """
    h, w = image_size
    segs = []

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Convert mask to numpy
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Ensure 2D
        while len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze(0) if mask_np.shape[0] == 1 else mask_np[0]

        # Resize if needed
        if mask_np.shape != (h, w):
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
            mask_np = torch.nn.functional.interpolate(
                mask_tensor, size=(h, w), mode="nearest"
            ).squeeze().numpy()

        # Get bounding box
        if i < len(boxes):
            box = boxes[i]
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
        else:
            # Calculate from mask
            rows = np.any(mask_np > 0.5, axis=1)
            cols = np.any(mask_np > 0.5, axis=0)
            if not rows.any() or not cols.any():
                continue
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

        # Ensure valid bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop mask
        cropped_mask = mask_np[y1:y2+1, x1:x2+1]
        cropped_mask_tensor = torch.from_numpy(cropped_mask).float()

        # Create SEG tuple
        seg = (
            cropped_mask_tensor,                    # cropped mask
            (x1, y1, x2 - x1, y2 - y1),            # crop_region (x, y, w, h)
            (x1, y1, x2, y2),                       # bbox (x1, y1, x2, y2)
            label,                                  # label
            float(score)                            # confidence
        )
        segs.append(seg)

    return ((w, h), segs)
