"""
ComfyUI SAM3 Nodes, unified model loader for both image and video using official Meta sam3_lib.
All class names and functions prefixed with TBG for uniqueness.
"""

import torch
from PIL import Image
import numpy as np

import json
import io
import base64

from sam3_utils import (
    comfy_image_to_pil,
    pil_to_comfy_image,
    masks_to_comfy_mask,
    visualize_masks_on_image,
    tensor_to_list,
    ensure_model_on_device,
    offload_model_if_needed,
    DepthEstimator,
)

from .sam3_lib.model_builder import build_sam3_image_model, build_sam3_video_predictor
from .sam3_lib.model.sam3_image_processor import Sam3Processor
from typing import Tuple, Optional


# Impact-Pack style MASK -> SEGS helper (your file in same folder)
from .masktosegs import mask_to_segs, SEG

_MODEL_CACHE = {}


from .model_manager import get_available_models, get_model_path, download_sam3_model
from .sam3_utils import SAM3ImageSegmenter
import os
try:
    import folder_paths
    base_models_folder = folder_paths.models_dir
except ImportError:
    base_models_folder = "models"




class TBGSAM3ModelLoaderAndDownloader:
    """
    Advanced SAM3 model loader that:
    - Can use the official API (auto configuration)
    - Can auto-download a local checkpoint if missing
    - Can load a specific local checkpoint under models/sam3
    Returns the same SAM3_MODEL dict as TBGLoadSAM3Model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # List known local models from model_manager
        # get_available_models() returns ["auto (download from HuggingFace)", <files...>]
        available = get_available_models()
        # Present clearer choices in UI
        model_sources = [
            "auto (API to cache)",            # build default model (no fixed ckpt path)
            "local (auto-download)", # download sam3.pt into models/sam3 if missing
        ] + available[1:]           # additional discovered checkpoint files

        return {
            "required": {
                "model_source": (model_sources, {"default": "local (auto-download)"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "TBG/SAM3"

    def load_model(self, model_source: str, device: str):
        hf_repo  = "facebook/sam3"

        """
        Build and return a SAM3_MODEL dict:
          {model, processor, device, original_device}
        """
        # Resolve checkpoint path if needed
        checkpoint_path = None

        if model_source == "auto (API to cache)":
            # Let builder construct its default weights / config
            print("[TBGSAM3ModelLoaderAdvanced] Using API/default SAM3 image model.")
            checkpoint_path = None

        elif model_source == "local (auto-download)":
            # Download only sam3.pt into models/sam3
            sam3_dir = download_sam3_model(hf_repo)     # returns models/sam3
            checkpoint_path = os.path.join(sam3_dir, "sam3.pt")
            if not os.path.isfile(checkpoint_path):
                raise RuntimeError(
                    f"[TBGSAM3ModelLoaderAdvanced] Downloaded model file not found at: {checkpoint_path}"
                )
            print(f"[TBGSAM3ModelLoaderAdvanced] Using downloaded local checkpoint: {checkpoint_path}")

        else:
            # Specific local checkpoint chosen from list under models/sam3
            checkpoint_path = get_model_path(model_source)
            if not checkpoint_path or not os.path.isfile(checkpoint_path):
                raise RuntimeError(
                    f"[TBGSAM3ModelLoaderAdvanced] Local model file not found: {model_source} -> {checkpoint_path}"
                )
            print(f"[TBGSAM3ModelLoaderAdvanced] Using selected local checkpoint: {checkpoint_path}")

        # --- Build SAM3 image model + processor, mirroring TBGLoadSAM3Model ---

        if checkpoint_path:
            sam3_model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        else:
            sam3_model = build_sam3_image_model()

        processor = Sam3Processor(sam3_model)

        sam3_model.to(device)
        sam3_model.processor = processor
        sam3_model.eval()

        model_dict = {
            "model": sam3_model,
            "processor": processor,
            "device": device,
            "original_device": device,
        }

        print("[TBGSAM3ModelLoaderAdvanced] SAM3 model ready on device:", device)
        return (model_dict,)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class TBGLoadSAM3Model:
    """
    Simple SAM3 loader using the new models/sam3 folder.

    Currently supports image mode only (no video).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    FUNCTION = "tbg_load_model"
    CATEGORY = "TBG/SAM3"

    def tbg_load_model(self, device: str):
        # Ensure base folder exists (models/sam3)
        _ = get_available_models()  # implicitly creates models/sam3 via model_manager

        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        model.to(device)
        model.processor = processor
        model.eval()

        model_dict = {
            "model": model,
            "processor": processor,
            "device": device,
            "original_device": device,
        }

        return (model_dict,)




class TBGSam3Segmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model loaded from LoadSAM3Model node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image to perform segmentation on"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Minimum confidence score to keep detections. Lower threshold (0.2) works better with SAM3's presence scoring"
                }),

                "pipeline_mode": (["all", "boxes_only", "points_only", "positive_only", "negative_only", "disabled"], {
                    "default": "all",
                    "tooltip": "Which prompts from pipeline to use."
                }),
                "detect_all": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Detect All",
                    "label_off": "Limit Detections to max_detection",
                    "tooltip": "When enabled, detects all objects. When disabled, uses max_detections value."
                }),
                "max_detections": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Maximum detections when detect_all is disabled."
                }),
                "instances": ("BOOLEAN", {
                    "default": False,
                    "label_on": "No Instances",
                    "label_off": "All Instances",
                    "tooltip": (
                        "When ON: keep only detections whose boxes overlap a positive box or contain a positive point.\n"
                        "When OFF: return all SAM3 detections including instances."
                    )
                }),
                "crop_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Crop factor used when building combined SEGS (Impact Pack style). 1.0 = tight bbox."
                }),
                "min_size": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 500,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Minimum segment size in pixels as a square side. 1=1x1, 200=200x200; smaller masks are discarded."
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Fill Holes",
                    "label_off": "Keep Holes",
                    "tooltip": "When enabled, fills holes inside each mask (solid segments)."
                }),

            },
            "optional": {
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "e.g., 'cat', 'person in red', 'car'",
                    "tooltip": "Text to guide segmentation (optional)."
                }),
                "sam3_selectors_pipe": ("SAM3_PROMPT_PIPELINE", {
                    "tooltip": "Unified pipeline containing boxes/points)."
                }),
                "mask_prompt": ("MASK", {
                    "tooltip": "Optional mask to refine the segmentation."
                }),

            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "STRING", "SEGS", "MASK", "SEGS")
    RETURN_NAMES = ("masks", "visualization", "boxes", "scores", "segs", "combined_mask", "combined_segs")
    FUNCTION = "segment"
    CATEGORY = "TBG/SAM3"

    def segment(self, sam3_model, image, confidence_threshold=0.2, detect_all=True,
                pipeline_mode="all", instances=False, crop_factor=1.5, min_size=32,
                fill_holes=False, text_prompt="", sam3_selectors_pipe=None,
                mask_prompt=None, exemplar_box=None, exemplar_mask=None,
                max_detections=10):

        actual_max_detections = -1 if detect_all else max_detections

        positive_boxes = None
        negative_boxes = None
        positive_points = None
        negative_points = None

        def _valid_block(block, key):
            return isinstance(block, dict) and key in block and bool(block[key])

        # --- Select prompts from unified pipeline (user input only) ---
        if sam3_selectors_pipe is not None and pipeline_mode != "disabled":
            if not isinstance(sam3_selectors_pipe, dict):
                raise ValueError(f"sam3_selectors_pipe must be a dictionary, got {type(sam3_selectors_pipe)}")

            pipeline_positive_boxes = sam3_selectors_pipe.get("positive_boxes", None)
            pipeline_negative_boxes = sam3_selectors_pipe.get("negative_boxes", None)
            pipeline_positive_points = sam3_selectors_pipe.get("positive_points", None)
            pipeline_negative_points = sam3_selectors_pipe.get("negative_points", None)

            print(
                "[SAM3] pipeline input: "
                f"pos_boxes={_valid_block(pipeline_positive_boxes, 'boxes')} "
                f"(len={len(pipeline_positive_boxes['boxes']) if _valid_block(pipeline_positive_boxes, 'boxes') else 0}), "
                f"neg_boxes={_valid_block(pipeline_negative_boxes, 'boxes')} "
                f"(len={len(pipeline_negative_boxes['boxes']) if _valid_block(pipeline_negative_boxes, 'boxes') else 0}), "
                f"pos_points={_valid_block(pipeline_positive_points, 'points')} "
                f"(len={len(pipeline_positive_points['points']) if _valid_block(pipeline_positive_points, 'points') else 0}), "
                f"neg_points={_valid_block(pipeline_negative_points, 'points')} "
                f"(len={len(pipeline_negative_points['points']) if _valid_block(pipeline_negative_points, 'points') else 0})"
            )

            if pipeline_mode == "all":
                positive_boxes = pipeline_positive_boxes
                negative_boxes = pipeline_negative_boxes
                positive_points = pipeline_positive_points
                negative_points = pipeline_negative_points
            elif pipeline_mode == "boxes_only":
                positive_boxes = pipeline_positive_boxes
                negative_boxes = pipeline_negative_boxes
            elif pipeline_mode == "points_only":
                positive_points = pipeline_positive_points
                negative_points = pipeline_negative_points
            elif pipeline_mode == "positive_only":
                positive_boxes = pipeline_positive_boxes
                positive_points = pipeline_positive_points
            elif pipeline_mode == "negative_only":
                negative_boxes = pipeline_negative_boxes
                negative_points = pipeline_negative_points

        print(
            f"[SAM3] pipeline_mode='{pipeline_mode}', instances={instances} | "
            f"pos_boxes_active={_valid_block(positive_boxes, 'boxes')}, "
            f"neg_boxes_active={_valid_block(negative_boxes, 'boxes')}, "
            f"pos_points_active={_valid_block(positive_points, 'points')}, "
            f"neg_points_active={_valid_block(negative_points, 'points')}"
        )

        # --- Setup model / processor ---
        ensure_model_on_device(sam3_model)
        processor = sam3_model["processor"]

        print(f"[SAM3] Running segmentation")
        print(f"[SAM3] Confidence threshold: {confidence_threshold}")

        pil_image = comfy_image_to_pil(image)
        print(f"[SAM3] Image size: {pil_image.size}")

        batch_size, height, width, channels = image.shape
        processor.set_confidence_threshold(confidence_threshold)

        state = processor.set_image(pil_image)

        # --- Apply prompts (user input) ---
        if text_prompt and text_prompt.strip():
            print(f"[SAM3] Using text_prompt='{text_prompt.strip()}'")
            state = processor.set_text_prompt(text_prompt.strip(), state)

        # Boxes from user pipeline
        all_boxes = []
        all_box_labels = []
        if _valid_block(positive_boxes, "boxes"):
            all_boxes.extend(positive_boxes["boxes"])
            all_box_labels.extend(positive_boxes.get("labels", [1] * len(positive_boxes["boxes"])))
        if _valid_block(negative_boxes, "boxes"):
            all_boxes.extend(negative_boxes["boxes"])
            all_box_labels.extend(negative_boxes.get("labels", [0] * len(negative_boxes["boxes"])))
        print(f"[SAM3] total box prompts={len(all_boxes)}")
        if all_boxes:
            state = processor.add_multiple_box_prompts(all_boxes, all_box_labels, state)

        # Points from user pipeline
        all_points = []
        all_point_labels = []
        if _valid_block(positive_points, "points"):
            all_points.extend(positive_points["points"])
            all_point_labels.extend(positive_points.get("labels", [1] * len(positive_points["points"])))
        if _valid_block(negative_points, "points"):
            all_points.extend(negative_points["points"])
            all_point_labels.extend(negative_points.get("labels", [0] * len(negative_points["points"])))
        print(f"[SAM3] total point prompts={len(all_points)}")
        if all_points:
            state = processor.add_point_prompt(all_points, all_point_labels, state)

        # Optional extra mask_prompt
        if mask_prompt is not None:
            if not isinstance(mask_prompt, torch.Tensor):
                mask_prompt = torch.from_numpy(mask_prompt)
            mask_prompt = mask_prompt.to(sam3_model["device"])
            print("[SAM3] Adding external mask_prompt")
            state = processor.add_mask_prompt(mask_prompt, state)

        # --- Run SAM3 ---
        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)

        total_scores = len(scores) if scores is not None else 0
        print(f"[SAM3 DEBUG] RAW PREDICTIONS: total {total_scores}")
        if boxes is not None:
            print(f"[SAM3 DEBUG] Output boxes shape: {boxes.shape}")

        # --- Filter out segments smaller than min_size x min_size ---
        if masks is not None and masks.numel() > 0 and min_size > 1:
            import torch as _torch

            # Convert side length to minimum area
            min_area = float(min_size * min_size)

            # Flatten masks to [N,H,W] for area computation
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks_flat = masks[:, 0, :, :]
            elif masks.dim() == 3:
                masks_flat = masks
            elif masks.dim() == 4:
                masks_flat = masks.mean(dim=1)
            else:
                raise ValueError(f"[SAM3] Unexpected masks shape for min_size filter: {masks.shape}")

            binary = (masks_flat > 0.5).float()
            areas = binary.view(binary.shape[0], -1).sum(dim=1)

            keep_indices = (areas >= min_area).nonzero(as_tuple=False).view(-1)
            print(f"[SAM3] min_size={min_size}px -> min_area={min_area} px, keeping {keep_indices.numel()} of {binary.shape[0]} masks")

            if keep_indices.numel() > 0:
                masks = masks[keep_indices]
                boxes = boxes[keep_indices] if boxes is not None else None
                scores = scores[keep_indices] if scores is not None else None
            else:
                print("[SAM3] All detections removed by min_size filter; returning empty result")
                h, w = pil_image.size[1], pil_image.size[0]
                empty_mask = _torch.zeros(1, h, w, device=masks.device)
                empty_segs = ((height, width), [])
                offload_model_if_needed(sam3_model)
                return (empty_mask, pil_to_comfy_image(pil_image), "[]", "[]", empty_segs, empty_mask, empty_segs)

        if masks is None or len(masks) == 0:
            print(f"[SAM3] No detections found at threshold {confidence_threshold}")
            h, w = pil_image.size[1], pil_image.size[0]
            empty_mask = torch.zeros(1, h, w)
            empty_segs = ((height, width), [])
            offload_model_if_needed(sam3_model)
            return (empty_mask, pil_to_comfy_image(pil_image), "[]", "[]", empty_segs, empty_mask, empty_segs)

        # --- Instance filtering using ONLY user positive prompts ---
        if instances and boxes is not None:
            print("[SAM3] Instances filter: keep only detections overlapping positive boxes / containing positive points")

            boxes_cpu = boxes.detach().cpu()
            print(f"[SAM3] Instances filter: total detections before filter={len(boxes_cpu)}")

            positive_prompt_boxes = []
            if _valid_block(positive_boxes, "boxes"):
                for idx, (cx, cy, w_norm, h_norm) in enumerate(positive_boxes["boxes"]):
                    x1 = (cx - w_norm / 2.0) * width
                    y1 = (cy - h_norm / 2.0) * height
                    x2 = (cx + w_norm / 2.0) * width
                    y2 = (cy + h_norm / 2.0) * height
                    positive_prompt_boxes.append([x1, y1, x2, y2])
                    print(f"[SAM3] pos_box[{idx}] norm=({cx:.3f},{cy:.3f},{w_norm:.3f},{h_norm:.3f}) -> px=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

            positive_prompt_points = []
            if _valid_block(positive_points, "points"):
                for idx, (px_norm, py_norm) in enumerate(positive_points["points"]):
                    px = px_norm * width
                    py = py_norm * height
                    positive_prompt_points.append([px, py])
                    print(f"[SAM3] pos_pt[{idx}] norm=({px_norm:.3f},{py_norm:.3f}) -> px=({px:.1f},{py:.1f})")

            keep_indices = []
            iou_threshold = 0.1

            for i, det_box in enumerate(boxes_cpu):
                db = det_box.tolist()
                ax1, ay1, ax2, ay2 = db
                print(f"[SAM3] det[{i}] box=({ax1:.1f},{ay1:.1f},{ax2:.1f},{ay2:.1f})")

                # IoU with each positive box
                max_iou = 0.0
                for j, pb in enumerate(positive_prompt_boxes):
                    bx1, by1, bx2, by2 = pb
                    ix1 = max(ax1, bx1)
                    iy1 = max(ay1, by1)
                    ix2 = min(ax2, bx2)
                    iy2 = min(ay2, by2)
                    iw = max(0.0, ix2 - ix1)
                    ih = max(0.0, iy2 - iy1)
                    inter = iw * ih
                    if inter > 0:
                        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
                        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
                        union = area_a + area_b - inter
                        if union > 0:
                            iou_val = inter / union
                            print(f"[SAM3]  det[{i}] vs pos_box[{j}] IoU={iou_val:.3f}")
                            if iou_val > max_iou:
                                max_iou = iou_val

                # Check if any positive point lies inside this detection box
                point_inside = False
                if positive_prompt_points:
                    for px, py in positive_prompt_points:
                        if ax1 <= px <= ax2 and ay1 <= py <= ay2:
                            point_inside = True
                            break

                keep = (positive_prompt_boxes and max_iou >= iou_threshold) or (
                    positive_prompt_points and point_inside
                )
                print(f"[SAM3]  det[{i}] point_inside={point_inside}, max_iou={max_iou:.3f}")

                if keep:
                    keep_indices.append(i)

            if keep_indices:
                keep = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
                masks = masks[keep]
                boxes = boxes[keep] if boxes is not None else None
                scores = scores[keep] if scores is not None else None
                print(f"[SAM3] Instances filter kept {len(keep_indices)} of {len(boxes_cpu)} detections")
            else:
                print("[SAM3] Instances filter removed all detections; returning empty result")
                h, w = pil_image.size[1], pil_image.size[0]
                empty_mask = torch.zeros(1, h, w, device=boxes.device if boxes is not None else "cpu")
                empty_segs = ((height, width), [])
                offload_model_if_needed(sam3_model)
                return (empty_mask, pil_to_comfy_image(pil_image), "[]", "[]", empty_segs, empty_mask, empty_segs)

        # --- Limit by max_detections ---
        if actual_max_detections > 0 and len(masks) > actual_max_detections:
            if scores is not None:
                top_indices = torch.argsort(scores, descending=True)[:actual_max_detections]
                masks = masks[top_indices]
                boxes = boxes[top_indices] if boxes is not None else None
                scores = scores[top_indices] if scores is not None else None

        # --- Optional: fill holes inside each mask (per-segment, safe) ---
        if fill_holes and isinstance(masks, torch.Tensor) and masks.numel() > 0:
            import cv2
            import numpy as np

            device = masks.device

            # Normalize to [N,H,W] float on CPU
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks_flat = masks[:, 0, :, :].detach().cpu()
            elif masks.dim() == 3:
                masks_flat = masks.detach().cpu()
            elif masks.dim() == 4:
                masks_flat = masks.mean(dim=1).detach().cpu()
            else:
                raise ValueError(f"[SAM3] Unexpected masks shape for fill_holes: {masks.shape}")

            filled_list = []
            for idx in range(masks_flat.shape[0]):
                m = masks_flat[idx].numpy()  # [H,W], float32

                # Binary foreground: 1 = segment, 0 = background
                fg = (m > 0.5).astype(np.uint8)
                if fg.sum() == 0:
                    filled_list.append(fg.astype(np.float32))
                    continue

                h, w = fg.shape

                # Tight bbox of the segment
                ys, xs = np.where(fg == 1)
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()

                crop_fg = fg[y1:y2 + 1, x1:x2 + 1]  # foreground inside bbox
                ch, cw = crop_fg.shape

                # Background inside bbox
                inv = 1 - crop_fg  # 1 = background inside bbox

                # Flood fill background from crop border to find outer background
                inv_ff = inv.copy()
                mask_ff = np.zeros((ch + 2, cw + 2), np.uint8)

                # Flood from all 4 corners of the crop
                cv2.floodFill(inv_ff, mask_ff, (0, 0), 2)
                cv2.floodFill(inv_ff, mask_ff, (cw - 1, 0), 2)
                cv2.floodFill(inv_ff, mask_ff, (0, ch - 1), 2)
                cv2.floodFill(inv_ff, mask_ff, (cw - 1, ch - 1), 2)

                # Outer background: inv_ff == 2
                outer_bg = (inv_ff == 2).astype(np.uint8)

                # Holes: background pixels not connected to border
                holes = inv - outer_bg
                holes[holes < 0] = 0

                # Fill holes into foreground
                filled_crop = crop_fg + holes
                filled_crop = np.clip(filled_crop, 0, 1).astype(np.uint8)

                # Put back into full-size mask
                filled_full = fg.copy()
                filled_full[y1:y2 + 1, x1:x2 + 1] = filled_crop

                filled_list.append(filled_full.astype(np.float32))

            filled_stack = torch.from_numpy(np.stack(filled_list, axis=0)).to(device)  # [N,H,W]

            # Restore original mask tensor shape
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = filled_stack.unsqueeze(1)  # [N,1,H,W]
            else:
                masks = filled_stack  # [N,H,W]

        # --- Build outputs ---
        comfy_masks = masks_to_comfy_mask(masks)

        # Combined full-image mask: union of all instance masks
        if isinstance(masks, torch.Tensor) and masks.numel() > 0:
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks_flat = masks[:, 0, :, :]
            elif masks.dim() == 3:
                masks_flat = masks
            elif masks.dim() == 4:
                masks_flat = masks.mean(dim=1)
            else:
                raise ValueError(f"[SAM3] Unexpected masks shape for combined mask: {masks.shape}")
            combined_tensor = (masks_flat > 0.5).any(dim=0, keepdim=True).float()
        else:
            h, w = pil_image.size[1], pil_image.size[0]
            combined_tensor = torch.zeros(1, h, w)

        combined_mask = masks_to_comfy_mask(combined_tensor)

        vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
        vis_tensor = pil_to_comfy_image(vis_image)

        def tensor_to_list_safe(t):
            if t is None:
                return []
            return tensor_to_list(t)

        boxes_list = tensor_to_list_safe(boxes)
        scores_list = tensor_to_list_safe(scores)

        boxes_json = json.dumps(boxes_list, indent=2)
        scores_json = json.dumps(scores_list, indent=2)

        # Per-instance SEGS (TBG format)
        segs = self._build_segs(
            masks=masks,
            boxes=boxes,
            scores=scores,
            original_image=image,
            text_prompt=text_prompt,
            width=width,
            height=height
        )

        #        # Impact-Pack style combined SEGS from combined mask using masktosegs
        from .masktosegs import make_2d_mask

        combined_label = text_prompt.strip() or "combined"

        # Ensure combined mask is on CPU and 2D before passing to mask_to_segs
        combined_cpu = combined_tensor.detach().cpu()          # [1,H,W] on CPU
        combined_2d = make_2d_mask(combined_cpu)               # [H,W] numpy

        combined_segs = mask_to_segs(
            combined_2d,
            combined=True,
            crop_factor=crop_factor,
            bbox_fill=False,
            drop_size=1,
            label=combined_label,
            crop_min_size=None,
            detailer_hook=None,
            is_contour=True
        )


        print(f"[SAM3] Segmentation complete. {len(comfy_masks)} masks, {len(segs[1])} SEGS, combined_segs has {len(combined_segs[1])} elements.")

        offload_model_if_needed(sam3_model)

        return (comfy_masks, vis_tensor, boxes_json, scores_json, segs, combined_mask, combined_segs)

    def _build_segs(self, masks, boxes, scores, original_image, text_prompt, width, height):
        """
        Build SEGS using the same logic as masktosegs.mask_to_segs, but per instance mask.

        Returns:
            ( (H, W), [SEG, SEG, ...] )
        """
        import numpy as np
        import torch
        from .masktosegs import make_2d_mask

        shape_info = (height, width)
        seg_list = []

        if masks is None or len(masks) == 0:
            return (shape_info, seg_list)

        # Ensure masks on CPU for numpy conversion
        if isinstance(masks, torch.Tensor):
            masks_cpu = masks.detach().cpu()
        else:
            masks_cpu = masks

        num_detections = len(masks_cpu)

        for i in range(num_detections):
            # Single instance mask: [H,W] or [1,H,W]
            mask_i = masks_cpu[i]

            # Convert to 2D numpy using the same helper as combined_segs
            mask_2d = make_2d_mask(mask_i)  # np.ndarray [H,W]

            # Optional: use a per-instance label if you like
            if text_prompt and text_prompt.strip():
                label = f"{text_prompt}_{i}"
            else:
                label = f"detection_{i}"

            # Use mask_to_segs with combined=False to split contours into SEGS
            # Use crop_factor=1.0 by default here; you can make it configurable if needed
            shape_inst, segs_inst = mask_to_segs(
                mask_2d,
                combined=False,
                crop_factor=1.0,
                bbox_fill=False,
                drop_size=1,
                label=label,
                crop_min_size=None,
                detailer_hook=None,
                is_contour=True
            )

            # segs_inst is a list of SEG instances; extend the global list
            if segs_inst:
                seg_list.extend(segs_inst)

        print(f"[SAM3] Built SEGS with {len(seg_list)} elements (via mask_to_segs per instance)")
        return (shape_info, seg_list)


class TBGSam3SegmentationBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {
                    "tooltip": "SAM3 model loaded from LoadSAM3Model node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image batch to perform segmentation on (B,H,W,C)"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Minimum confidence score to keep detections"
                }),
                "pipeline_mode": (["all", "boxes_only", "points_only", "positive_only", "negative_only", "disabled"], {
                    "default": "all",
                    "tooltip": "Which prompts from pipeline to use"
                }),
                "detect_all": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Detect All",
                    "label_off": "Limit Detections",
                    "tooltip": "When enabled, detects all objects. When disabled, uses max_detections"
                }),
                "max_detections": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Maximum detections when detect_all is disabled"
                }),
                "instances": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Filter Instances",
                    "label_off": "All Instances",
                    "tooltip": "When ON: keep only detections overlapping positive prompts"
                }),
                "crop_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Crop factor for SEGS. 1.0 = tight bbox"
                }),
                "min_size": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 500,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Minimum segment size (square side in pixels)"
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Fill Holes",
                    "label_off": "Keep Holes",
                    "tooltip": "Fill holes inside masks"
                }),
            },
            "optional": {
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "e.g., 'cat', 'person in red', 'car'",
                    "tooltip": "Text to guide segmentation"
                }),
                "sam3_selectors_pipe": ("SAM3_PROMPT_PIPELINE", {
                    "tooltip": "Unified pipeline containing boxes/points"
                }),
                "mask_prompt": ("MASK", {
                    "tooltip": "Optional batch mask (B,H,W or 1,H,W)"
                }),
            }
        }

    # Impact Pack compatible outputs
    RETURN_TYPES = ("SEGS", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("segs", "visualization", "combined_mask", "batch_masks")
    FUNCTION = "segment"
    CATEGORY = "TBG/SAM3"

    def segment(
            self,
            sam3_model,
            image,
            confidence_threshold=0.4,
            detect_all=True,
            pipeline_mode="all",
            instances=False,
            crop_factor=1.5,
            min_size=100,
            fill_holes=False,
            text_prompt="",
            sam3_selectors_pipe=None,
            mask_prompt=None,
            max_detections=50,
    ):
        actual_max_detections = -1 if detect_all else max_detections
        batch_size, height, width, channels = image.shape

        def _valid_block(block, key):
            return isinstance(block, dict) and key in block and bool(block[key])

        # --- Extract prompts from pipeline ---
        positive_boxes = None
        negative_boxes = None
        positive_points = None
        negative_points = None

        if sam3_selectors_pipe is not None and pipeline_mode != "disabled":
            if not isinstance(sam3_selectors_pipe, dict):
                raise ValueError(f"sam3_selectors_pipe must be dict, got {type(sam3_selectors_pipe)}")

            pipeline_positive_boxes = sam3_selectors_pipe.get("positive_boxes", None)
            pipeline_negative_boxes = sam3_selectors_pipe.get("negative_boxes", None)
            pipeline_positive_points = sam3_selectors_pipe.get("positive_points", None)
            pipeline_negative_points = sam3_selectors_pipe.get("negative_points", None)

            if pipeline_mode == "all":
                positive_boxes = pipeline_positive_boxes
                negative_boxes = pipeline_negative_boxes
                positive_points = pipeline_positive_points
                negative_points = pipeline_negative_points
            elif pipeline_mode == "boxes_only":
                positive_boxes = pipeline_positive_boxes
                negative_boxes = pipeline_negative_boxes
            elif pipeline_mode == "points_only":
                positive_points = pipeline_positive_points
                negative_points = pipeline_negative_points
            elif pipeline_mode == "positive_only":
                positive_boxes = pipeline_positive_boxes
                positive_points = pipeline_positive_points
            elif pipeline_mode == "negative_only":
                negative_boxes = pipeline_negative_boxes
                negative_points = pipeline_negative_points

        print(f"[SAM3] Batch size: {batch_size}, pipeline_mode: {pipeline_mode}, instances: {instances}")

        # --- Accumulators for batch processing ---
        all_detection_masks = []  # All individual detection masks across all frames
        all_combined_masks = []  # One combined mask per frame
        all_vis_tensors = []  # One visualization per frame
        all_segs = []  # SEGS from all frames

        ensure_model_on_device(sam3_model)
        processor = sam3_model["processor"]
        processor.set_confidence_threshold(confidence_threshold)

        # --- Process each frame ---
        for b in range(batch_size):
            print(f"[SAM3] Frame {b + 1}/{batch_size}")

            img_tensor = image[b:b + 1]
            pil_image = comfy_image_to_pil(img_tensor)
            if isinstance(pil_image, list):
                pil_image_single = pil_image[0]
            else:
                pil_image_single = pil_image

            state = processor.set_image(pil_image_single)

            # Apply text prompt
            if text_prompt and text_prompt.strip():
                state = processor.set_text_prompt(text_prompt.strip(), state)

            # Apply box prompts
            all_boxes = []
            all_box_labels = []
            if _valid_block(positive_boxes, "boxes"):
                all_boxes.extend(positive_boxes["boxes"])
                all_box_labels.extend(positive_boxes.get("labels", [1] * len(positive_boxes["boxes"])))
            if _valid_block(negative_boxes, "boxes"):
                all_boxes.extend(negative_boxes["boxes"])
                all_box_labels.extend(negative_boxes.get("labels", [0] * len(negative_boxes["boxes"])))
            if all_boxes:
                state = processor.add_multiple_box_prompts(all_boxes, all_box_labels, state)

            # Apply point prompts
            all_points = []
            all_point_labels = []
            if _valid_block(positive_points, "points"):
                all_points.extend(positive_points["points"])
                all_point_labels.extend(positive_points.get("labels", [1] * len(positive_points["points"])))
            if _valid_block(negative_points, "points"):
                all_points.extend(negative_points["points"])
                all_point_labels.extend(negative_points.get("labels", [0] * len(negative_points["points"])))
            if all_points:
                state = processor.add_point_prompt(all_points, all_point_labels, state)

            # Apply mask prompt
            current_mask_prompt = None
            if mask_prompt is not None:
                if isinstance(mask_prompt, torch.Tensor):
                    if mask_prompt.dim() == 3 and mask_prompt.shape[0] == batch_size:
                        current_mask_prompt = mask_prompt[b]
                    else:
                        current_mask_prompt = mask_prompt
                else:
                    current_mask_prompt = mask_prompt

                if current_mask_prompt is not None:
                    if not isinstance(current_mask_prompt, torch.Tensor):
                        current_mask_prompt = torch.from_numpy(current_mask_prompt)
                    current_mask_prompt = current_mask_prompt.to(sam3_model["device"])
                    state = processor.add_mask_prompt(current_mask_prompt, state)

            # Run SAM3
            masks = state.get("masks", None)
            boxes = state.get("boxes", None)
            scores = state.get("scores", None)

            # Filter by min_size
            if masks is not None and isinstance(masks, torch.Tensor) and masks.numel() > 0 and min_size > 1:
                min_area = float(min_size * min_size)
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks_flat = masks[:, 0, :, :]
                elif masks.dim() == 3:
                    masks_flat = masks
                elif masks.dim() == 4:
                    masks_flat = masks.mean(dim=1)
                else:
                    raise ValueError(f"[SAM3] Unexpected masks shape: {masks.shape}")

                binary = (masks_flat > 0.5).float()
                areas = binary.view(binary.shape[0], -1).sum(dim=1)
                keep_indices = (areas >= min_area).nonzero(as_tuple=False).view(-1)

                if keep_indices.numel() > 0:
                    masks = masks[keep_indices]
                    boxes = boxes[keep_indices] if boxes is not None else None
                    scores = scores[keep_indices] if scores is not None else None
                else:
                    masks = None

            # Handle empty detections
            if masks is None or (isinstance(masks, torch.Tensor) and masks.numel() == 0):
                print(f"[SAM3] No detections for frame {b}")
                empty_combined = torch.zeros(1, height, width, device="cpu")
                all_combined_masks.append(empty_combined)
                all_vis_tensors.append(pil_to_comfy_image(pil_image_single))
                continue

            # Instance filtering
            if instances and boxes is not None and isinstance(boxes, torch.Tensor):
                boxes_cpu = boxes.detach().cpu()
                positive_prompt_boxes = []
                if _valid_block(positive_boxes, "boxes"):
                    for cx, cy, w_norm, h_norm in positive_boxes["boxes"]:
                        x1 = (cx - w_norm / 2.0) * width
                        y1 = (cy - h_norm / 2.0) * height
                        x2 = (cx + w_norm / 2.0) * width
                        y2 = (cy + h_norm / 2.0) * height
                        positive_prompt_boxes.append([x1, y1, x2, y2])

                positive_prompt_points = []
                if _valid_block(positive_points, "points"):
                    for px_norm, py_norm in positive_points["points"]:
                        positive_prompt_points.append([px_norm * width, py_norm * height])

                keep_indices = []
                iou_threshold = 0.1

                for i_det, det_box in enumerate(boxes_cpu):
                    ax1, ay1, ax2, ay2 = det_box.tolist()
                    max_iou = 0.0
                    for bx1, by1, bx2, by2 in positive_prompt_boxes:
                        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        if inter > 0:
                            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                            union = area_a + area_b - inter
                            if union > 0:
                                max_iou = max(max_iou, inter / union)

                    point_inside = any(ax1 <= px <= ax2 and ay1 <= py <= ay2
                                       for px, py in positive_prompt_points)

                    if (positive_prompt_boxes and max_iou >= iou_threshold) or \
                            (positive_prompt_points and point_inside):
                        keep_indices.append(i_det)

                if keep_indices:
                    keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
                    masks = masks[keep_tensor]
                    boxes = boxes[keep_tensor]
                    scores = scores[keep_tensor] if scores is not None else None
                else:
                    masks = None

            # Limit detections
            if isinstance(masks, torch.Tensor) and masks is not None and \
                    actual_max_detections > 0 and len(masks) > actual_max_detections:
                if scores is not None:
                    top_indices = torch.argsort(scores, descending=True)[:actual_max_detections]
                    masks = masks[top_indices]
                    boxes = boxes[top_indices]
                    scores = scores[top_indices]

            # Fill holes
            if fill_holes and isinstance(masks, torch.Tensor) and masks.numel() > 0:
                import cv2
                device = masks.device
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks_flat = masks[:, 0, :, :].detach().cpu()
                elif masks.dim() == 3:
                    masks_flat = masks.detach().cpu()
                else:
                    masks_flat = masks.mean(dim=1).detach().cpu()

                filled_list = []
                for idx_m in range(masks_flat.shape[0]):
                    m = masks_flat[idx_m].numpy()
                    fg = (m > 0.5).astype(np.uint8)
                    if fg.sum() == 0:
                        filled_list.append(fg.astype(np.float32))
                        continue

                    ys, xs = np.where(fg == 1)
                    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
                    crop_fg = fg[y1:y2 + 1, x1:x2 + 1]
                    ch, cw = crop_fg.shape

                    inv = 1 - crop_fg
                    inv_ff = inv.copy()
                    mask_ff = np.zeros((ch + 2, cw + 2), np.uint8)
                    cv2.floodFill(inv_ff, mask_ff, (0, 0), 2)
                    cv2.floodFill(inv_ff, mask_ff, (cw - 1, 0), 2)
                    cv2.floodFill(inv_ff, mask_ff, (0, ch - 1), 2)
                    cv2.floodFill(inv_ff, mask_ff, (cw - 1, ch - 1), 2)

                    outer_bg = (inv_ff == 2).astype(np.uint8)
                    holes = np.clip(inv - outer_bg, 0, 1)
                    filled_crop = np.clip(crop_fg + holes, 0, 1).astype(np.uint8)
                    filled_full = fg.copy()
                    filled_full[y1:y2 + 1, x1:x2 + 1] = filled_crop
                    filled_list.append(filled_full.astype(np.float32))

                filled_stack = torch.from_numpy(np.stack(filled_list, axis=0)).to(device)
                masks = filled_stack.unsqueeze(1) if masks.dim() == 4 and masks.shape[1] == 1 else filled_stack

            # Build per-frame outputs
            if masks is not None and isinstance(masks, torch.Tensor) and masks.numel() > 0:
                # Individual detection masks for this frame
                frame_detection_masks = masks_to_comfy_mask(masks)  # [N, H, W]
                all_detection_masks.append(frame_detection_masks)

                # Combined mask (union of all detections)
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks_flat = masks[:, 0, :, :]
                elif masks.dim() == 3:
                    masks_flat = masks
                else:
                    masks_flat = masks.mean(dim=1)

                combined_tensor = (masks_flat > 0.5).any(dim=0, keepdim=True).float()  # [1, H, W]
                all_combined_masks.append(combined_tensor)

                # Build SEGS for this frame
                segs = self._build_segs(
                    masks=masks,
                    boxes=boxes,
                    scores=scores,
                    original_image=img_tensor,
                    text_prompt=text_prompt,
                    width=width,
                    height=height,
                    crop_factor=crop_factor
                )
                all_segs.append(segs)

                # Visualization
                vis_image = visualize_masks_on_image(pil_image_single, masks, boxes, scores, alpha=0.5)
                all_vis_tensors.append(pil_to_comfy_image(vis_image))
            else:
                # Empty frame
                empty_combined = torch.zeros(1, height, width, device="cpu")
                all_combined_masks.append(empty_combined)
                all_vis_tensors.append(pil_to_comfy_image(pil_image_single))

        # --- Merge outputs ---
        # batch_masks: all detection masks from all frames [N_total, H, W]
        if all_detection_masks:
            batch_masks = torch.cat(all_detection_masks, dim=0)
        else:
            batch_masks = torch.zeros(0, height, width, device="cpu")

        # combined_mask: one per frame [B, H, W]
        combined_mask = torch.cat(all_combined_masks, dim=0)

        # visualization: one per frame [B, H, W, C]
        visualization = torch.cat(all_vis_tensors, dim=0)

        # SEGS: merge all frames into single SEGS
        if all_segs:
            shape_info = all_segs[0][0]
            merged_seg_list = []
            for shape_f, seg_list_f in all_segs:
                merged_seg_list.extend(seg_list_f)
            final_segs = (shape_info, merged_seg_list)
        else:
            final_segs = ((height, width), [])

        print(f"[SAM3] Complete: {len(final_segs[1])} SEGS, {batch_masks.shape[0]} detection masks, "
              f"{combined_mask.shape[0]} frames")

        offload_model_if_needed(sam3_model)

        return (final_segs, visualization, combined_mask, batch_masks)

    def _build_segs(self, masks, boxes, scores, original_image, text_prompt, width, height, crop_factor):
        from .masktosegs import make_2d_mask, mask_to_segs

        shape_info = (height, width)
        seg_list = []

        if masks is None or len(masks) == 0:
            return (shape_info, seg_list)

        masks_cpu = masks.detach().cpu() if isinstance(masks, torch.Tensor) else masks

        for i in range(len(masks_cpu)):
            mask_i = masks_cpu[i]
            mask_2d = make_2d_mask(mask_i)
            label = f"{text_prompt}_{i}" if text_prompt and text_prompt.strip() else f"detection_{i}"

            shape_inst, segs_inst = mask_to_segs(
                mask_2d,
                combined=False,
                crop_factor=crop_factor,
                bbox_fill=False,
                drop_size=1,
                label=label,
                crop_min_size=None,
                detailer_hook=None,
                is_contour=True,
            )
            if segs_inst:
                seg_list.extend(segs_inst)

        return (shape_info, seg_list)


class TBGSAM3PromptCollector:
    """
    Unified SAM3 Prompt Collector - collects points and boxes in single node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image for interactive selection. Use B to toggle Point/Box. Left=Positive, Right/Shift=Negative."
                }),
                "positive_points": ("STRING", {"default": "[]", "multiline": False}),
                "negative_points": ("STRING", {"default": "[]", "multiline": False}),
                "positive_boxes": ("STRING", {"default": "[]", "multiline": False}),
                "negative_boxes": ("STRING", {"default": "[]", "multiline": False}),
            }
        }


    RETURN_TYPES = ("SAM3_PROMPT_PIPELINE",)
    RETURN_NAMES = ("sam3_selectors_pipe",)
    FUNCTION = "collect_pipeline"
    CATEGORY = "TBG/SAM3"
    OUTPUT_NODE = True

    def collect_pipeline(self, image, positive_points, negative_points, positive_boxes, negative_boxes):
        # Parse JSON inputs
        try:
            pos_pts = json.loads(positive_points) if positive_points else []
            neg_pts = json.loads(negative_points) if negative_points else []
            pos_bxs = json.loads(positive_boxes) if positive_boxes else []
            neg_bxs = json.loads(negative_boxes) if negative_boxes else []
        except Exception:
            pos_pts, neg_pts, pos_bxs, neg_bxs = [], [], [], []

        print(f"[TBGSAM3PromptCollector] Points: +{len(pos_pts)} -{len(neg_pts)}, Boxes: +{len(pos_bxs)} -{len(neg_bxs)}")

        # Get image dimensions
        img_height, img_width = image.shape[1], image.shape[2]

        pipeline = {
            "positive_points": None,
            "negative_points": None,
            "positive_boxes": None,
            "negative_boxes": None
        }

        def normalize_points(pts):
            return [[p["x"] / img_width, p["y"] / img_height] for p in pts]

        if pos_pts:
            pipeline["positive_points"] = {
                "points": normalize_points(pos_pts),
                "labels": [1] * len(pos_pts),
            }

        if neg_pts:
            pipeline["negative_points"] = {
                "points": normalize_points(neg_pts),
                "labels": [0] * len(neg_pts),
            }

        def convert_boxes(boxes):
            converted = []
            for b in boxes:
                x1 = b["x1"] / img_width
                y1 = b["y1"] / img_height
                x2 = b["x2"] / img_width
                y2 = b["y2"] / img_height
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                converted.append([cx, cy, w, h])
            return converted

        if pos_bxs:
            pipeline["positive_boxes"] = {
                "boxes": convert_boxes(pos_bxs),
                "labels": [True] * len(pos_bxs),
            }

        if neg_bxs:
            pipeline["negative_boxes"] = {
                "boxes": convert_boxes(neg_bxs),
                "labels": [False] * len(neg_bxs),
            }

        # Convert image to base64 string for widget background
        img_tensor = image[0]
        if isinstance(img_tensor, torch.Tensor):
            img_array = img_tensor.detach().cpu().numpy()
        else:
            img_array = np.asarray(img_tensor)
        img_array = np.clip(img_array, 0.0, 1.0)
        img_array = (img_array * 255).astype(np.uint8)

        pil_img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "ui": {"bg_image": [img_base64]},
            "bg_image": [img_base64],
            "result": (pipeline,),
        }

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



NODE_CLASS_MAPPINGS = {
    "TBGLoadSAM3Model": TBGLoadSAM3Model,              # your simple loader
    "TBGSAM3ModelLoaderAdvanced": TBGSAM3ModelLoaderAndDownloader,  # new advanced loader
    "TBGSam3Segmentation": TBGSam3Segmentation,
    "TBGSam3SegmentationBatch": TBGSam3SegmentationBatch,
    "TBGSAM3PromptCollector": TBGSAM3PromptCollector,
    "SAM3DepthMap": SAM3DepthMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TBGLoadSAM3Model": "TBG SAM3 Model Loader",
    "TBGSAM3ModelLoaderAdvanced": "TBG SAM3 Model Loader and Downloader",
    "TBGSam3Segmentation": "TBG SAM3 Segmentation",
    "TBGSAM3PromptCollector": "TBG SAM3 Selector",
    "TBGSam3SegmentationBatch":"TBG SAM3 Batch Selector",
    "SAM3DepthMap": "SAM3 Depth Map",
}
