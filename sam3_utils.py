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

            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor


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

    def segment_with_points(
            self,
            image,
            points: List[Tuple[int, int]],
            point_labels: Optional[List[int]] = None,
    ):
        if point_labels is None:
            point_labels = [1] * len(points)

        # Ensure PIL
        if isinstance(image, torch.Tensor):
            pil_image = tensor_to_pil(image)
        else:
            pil_image = image

        # SAM3 expects 4D nested lists: [batch][num_objs][num_points][2]
        input_points = [[[list(p) for p in points]]]  # shape conceptually [1, 1, N, 2]
        input_labels = [[[int(l) for l in point_labels]]]  # shape [1, 1, N]

        inputs = self.processor(
            images=pil_image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post‑process; SAM3 examples typically use target_sizes / original_sizes from inputs
        results = self.processor.post_process_instance_segmentation(
            outputs=outputs,
            threshold=0.1,
            mask_threshold=0.1,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results["masks"]  # [num_inst, H, W]
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])

        return masks, boxes, scores




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

def extract_points_from_mask(mask: torch.Tensor, num_points: int = 5) -> List[Tuple[int, int]]:
    """Convert a binary/soft mask into a list of (x, y) points."""
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

"""
Utility functions for ComfyUI-SAM3 nodes
"""
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path


def get_comfy_models_dir():
    """Get the ComfyUI models directory"""
    # Try to find ComfyUI root by going up from custom_nodes
    current = Path(__file__).parent.parent.absolute()  # ComfyUI-SAM3
    comfy_custom_nodes = current.parent  # custom_nodes
    comfy_root = comfy_custom_nodes.parent  # ComfyUI root

    models_dir = comfy_root / "models" / "sam3"
    models_dir.mkdir(parents=True, exist_ok=True)

    return str(models_dir)


def comfy_image_to_pil(image):
    """
    Convert ComfyUI image tensor to PIL Image

    Args:
        image: ComfyUI image tensor [B, H, W, C] in range [0, 1]

    Returns:
        PIL Image
    """
    # ComfyUI images are [B, H, W, C] in range [0, 1]
    if isinstance(image, torch.Tensor):
        # Take first image if batch
        if image.dim() == 4:
            image = image[0]

        # Convert to numpy
        img_np = image.cpu().numpy()

        # Convert from [0, 1] to [0, 255]
        img_np = (img_np * 255).astype(np.uint8)

        # Convert to PIL
        pil_image = Image.fromarray(img_np)
        return pil_image

    return image


def pil_to_comfy_image(pil_image):
    """
    Convert PIL Image to ComfyUI image tensor

    Args:
        pil_image: PIL Image

    Returns:
        ComfyUI image tensor [1, H, W, C] in range [0, 1]
    """
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Convert to numpy array
    img_np = np.array(pil_image).astype(np.float32)

    # Normalize to [0, 1]
    img_np = img_np / 255.0

    # Convert to tensor [H, W, C]
    img_tensor = torch.from_numpy(img_np)

    # Add batch dimension [1, H, W, C]
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def masks_to_comfy_mask(masks):
    """
    Convert SAM3 masks to ComfyUI mask format

    Args:
        masks: torch.Tensor [N, H, W] or [N, 1, H, W] binary masks

    Returns:
        ComfyUI mask tensor [N, H, W] in range [0, 1] on CPU
    """
    if isinstance(masks, torch.Tensor):
        # Ensure float type and range [0, 1]
        masks = masks.float()
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present (N, 1, H, W) -> (N, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        # Move to CPU to ensure compatibility with downstream nodes
        return masks.cpu()
    elif isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks).float()
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        # Already on CPU since from numpy
        return masks

    return masks


def visualize_masks_on_image(image, masks, boxes=None, scores=None, alpha=0.5):
    """
    Create visualization of masks overlaid on image

    Args:
        image: PIL Image or numpy array
        masks: torch.Tensor [N, H, W] binary masks
        boxes: Optional torch.Tensor [N, 4] bounding boxes in [x0, y0, x1, y1]
        scores: Optional torch.Tensor [N] confidence scores
        alpha: Transparency of mask overlay

    Returns:
        PIL Image with visualization
    """
    if isinstance(image, torch.Tensor):
        image = comfy_image_to_pil(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))

    # Convert to numpy for processing
    img_np = np.array(image).astype(np.float32) / 255.0

    # Resize masks to image size if needed
    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = masks

    # Create colored overlay
    np.random.seed(42)  # Consistent colors
    overlay = img_np.copy()

    for i, mask in enumerate(masks_np):
        # Squeeze extra dimensions (masks may be [1, H, W] or [H, W])
        while mask.ndim > 2:
            mask = mask.squeeze(0)

        # Resize mask to image size if needed
        if mask.shape != img_np.shape[:2]:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((img_np.shape[1], img_np.shape[0]), PILImage.NEAREST)
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        # Random color for this mask
        color = np.random.rand(3)

        # Apply colored mask
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask > 0.5,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

    # Convert back to PIL
    result = Image.fromarray((overlay * 255).astype(np.uint8))

    # Draw boxes if provided
    if boxes is not None:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result)

        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes

        for i, box in enumerate(boxes_np):
            x0, y0, x1, y1 = box

            # Random color for this box (same seed for consistency)
            np.random.seed(42 + i)
            color_int = tuple((np.random.rand(3) * 255).astype(int).tolist())

            # Draw box
            draw.rectangle([x0, y0, x1, y1], outline=color_int, width=3)

            # Draw score if provided
            if scores is not None:
                score = scores[i] if isinstance(scores, (list, np.ndarray)) else scores[i].item()
                text = f"{score:.2f}"
                draw.text((x0, y0 - 15), text, fill=color_int)

    return result


def tensor_to_list(tensor):
    """Convert torch tensor to python list"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().tolist()
    return tensor


def ensure_model_on_device(sam3_model, target_device=None):
    """
    Ensure model is on the target device before inference

    Args:
        sam3_model: Model dict from LoadSAM3Model
        target_device: Target device (uses original_device if None)

    Returns:
        None (modifies model dict in place)
    """
    model = sam3_model["model"]
    processor = sam3_model["processor"]

    if target_device is None:
        target_device = sam3_model["original_device"]

    # Check if model is already on target device
    current_device = next(model.parameters()).device
    if str(current_device) != target_device:
        print(f"[SAM3] Moving model from {current_device} to {target_device}")
        model.to(target_device)
        processor.device = target_device
        sam3_model["device"] = target_device


def offload_model_if_needed(sam3_model):
    """
    Offload model to CPU if use_gpu_cache is False

    Args:
        sam3_model: Model dict from LoadSAM3Model

    Returns:
        None (modifies model dict in place)
    """
    use_gpu_cache = sam3_model.get("use_gpu_cache", True)

    if not use_gpu_cache:
        model = sam3_model["model"]
        processor = sam3_model["processor"]
        current_device = next(model.parameters()).device

        # Only offload if currently on GPU
        if "cuda" in str(current_device):
            print(f"[SAM3] Offloading model to CPU to free VRAM")
            model.to("cpu")
            processor.device = "cpu"
            sam3_model["device"] = "cpu"
            # Force garbage collection to free VRAM
            torch.cuda.empty_cache()
            import gc
            gc.collect()
