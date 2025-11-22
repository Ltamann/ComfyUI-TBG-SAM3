# MaskToSEGS Node for ComfyUI
# Copyright (C) 2025 TBG
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections import namedtuple

import cv2
import numpy as np
import torch

MAX_RESOLUTION = 4096  # Or set as needed

SEG = namedtuple("SEG",
    ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
    defaults=[None]
)


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)
    return int(new_startp), int(new_endp)

def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    crop_w = max(bbox_w+72, crop_w)
    crop_h = max(bbox_h+72, crop_h)

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)
    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2
    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)
    return [new_x1, new_y1, new_x2, new_y2]

def make_2d_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = np.squeeze(mask)
    if mask.ndim > 2:
        mask = mask[0]
    return mask

def mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size=1, label='A', crop_min_size=None, detailer_hook=None, is_contour=True):
    drop_size = max(drop_size, 1)
    if mask is None:
        print("[mask_to_segs] Cannot operate: MASK is empty.")
        return ([],)
    if isinstance(mask, np.ndarray):
        pass
    else:
        try:
            mask = mask.numpy()
        except AttributeError:
            print("[mask_to_segs] Cannot operate: MASK is not a NumPy array or Tensor.")
            return ([],)
    if mask is None:
        print("[mask_to_segs] Cannot operate: MASK is empty.")
        return ([],)
    result = []

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)
    for i in range(mask.shape[0]):
        mask_i = mask[i]
        if combined:
            indices = np.nonzero(mask_i)
            if len(indices[0]) > 0 and len(indices[1]) > 0:
                bbox = (
                    np.min(indices[1]),
                    np.min(indices[0]),
                    np.max(indices[1]),
                    np.max(indices[0]),
                )
                crop_region = make_crop_region(
                    mask_i.shape[1], mask_i.shape[0], bbox, crop_factor
                )
                x1, y1, x2, y2 = crop_region
                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_region)
                if x2 - x1 > 0 and y2 - y1 > 0:
                    cropped_mask = mask_i[y1:y2, x1:x2]
                    if bbox_fill:
                        bx1, by1, bx2, by2 = bbox
                        cropped_mask = cropped_mask.copy()
                        cropped_mask[by1:by2, bx1:bx2] = 1.0
                    if cropped_mask is not None:
                        item = SEG(
                            cropped_image=None,
                            cropped_mask=cropped_mask,
                            confidence=1.0,
                            crop_region=crop_region,
                            bbox=bbox,
                            label=label,
                            control_net_wrapper=None
                        )
                        result.append(item)
        else:
            mask_i_uint8 = (mask_i * 255.0).astype(np.uint8)
            try:
                # OpenCV 4.x style
                contours, ctree = cv2.findContours(mask_i_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except ValueError:
                # OpenCV 3.x style
                _, contours, ctree = cv2.findContours(mask_i_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #contours, ctree = cv2.findContours(mask_i_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if ctree is None or len(contours) == 0:
                continue
            for j, contour in enumerate(contours):
                hierarchy = ctree[0][j]
                if hierarchy[3] != -1:
                    continue
                separated_mask = np.zeros_like(mask_i_uint8)
                cv2.drawContours(separated_mask, [contour], 0, 255, -1)
                separated_mask = np.array(separated_mask / 255.0).astype(np.float32)
                x, y, w, h = cv2.boundingRect(contour)
                bbox = x, y, x + w, y + h
                crop_region = make_crop_region(
                    mask_i.shape[1], mask_i.shape[0], bbox, crop_factor, crop_min_size
                )
                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_region)
                if w > drop_size and h > drop_size:
                    if is_contour:
                        mask_src = separated_mask
                    else:
                        mask_src = mask_i * separated_mask
                    cropped_mask = np.array(
                        mask_src[
                            crop_region[1]: crop_region[3],
                            crop_region[0]: crop_region[2],
                        ]
                    )
                    if bbox_fill:
                        cx1, cy1, _, _ = crop_region
                        bx1 = x - cx1
                        bx2 = x+w - cx1
                        by1 = y - cy1
                        by2 = y+h - cy1
                        cropped_mask[by1:by2, bx1:bx2] = 1.0
                    if cropped_mask is not None:
                        cropped_mask = torch.clip(torch.from_numpy(cropped_mask), 0, 1.0).numpy()
                        item = SEG(
                            cropped_image=None,
                            cropped_mask=cropped_mask,
                            confidence=1.0,
                            crop_region=crop_region,
                            bbox=bbox,
                            label=label,
                            control_net_wrapper=None
                        )
                        result.append(item)
    if not result:
        print(f"TBG Masked Attention Empty mask.")
    else:
        print(f"TBG Masked Attention of Detected SEGS: {len(result)}")
    return (mask.shape[1], mask.shape[2]), result

@staticmethod
def MaskToSEGS(mask, combined, crop_factor, bbox_fill, drop_size, contour_fill=False):
    mask = make_2d_mask(mask)
    return mask_to_segs(
        mask,
        combined,
        crop_factor,
        bbox_fill,
        drop_size,
        is_contour=contour_fill
    )


