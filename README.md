**Updated Release: ComfyUI-TBG-SAM3 — Now we can plug a cleaned-up SAM3 segment straight into TBG Enhanced Refiner or any SEGS-ready input, like the Impact Pack effortlessly! So whats new.**

# ComfyUI-TBG-SAM3

A ComfyUI custom node extension integrating Meta's **Segment Anything Model 3 (SAM 3)** for advanced image and video segmentation capabilities. This extension provides production-ready nodes compatible with ComfyUI’s Impact Pack SEGS format, enabling text-prompt, point-based, and mask-driven segmentation as well as depth map generation per segment or for full images.
A ComfyUI custom-node extension that integrates Meta’s **Segment Anything Model 3 (SAM-3)** for advanced image segmentation. It supports unified point and box selection, text-prompt segmentation, point-guided masks, and mask-driven refinement. Features include an **Instant Instance Selection toggle**, **compatibility with tile-based upscalers such as TBG-ETUR**, and full support for **Impact Pack SEGS formats**.

The ComfyUI-TBG-SAM3 update focuses on making SAM3 segmentation easier to use, more compatible with common workflows, and cleaner in its final output. The node set still uses the same three core nodes — TBG SAM3 ModelLoader & Downloader, TBG SAM3 Segmentation, and TBG SAM3 Selector — but each one has been improved.

-**Key Improvements**

-**Unified Point-and-Box Selector**
The TBG SAM3 Selector now combines point and box selection into a single, streamlined tool. It supports both positive and negative prompts and keeps everything connected cleanly, making interactive segmentation much easier.

-**Enhanced Segmentation Logic + Instance Toggle**
The TBG SAM3 Segmentation node now uses the official SAM3 segmentation workflow and includes a new switch that lets you turn off instance generation when you don’t need it.

-**Tile-Based TBG-ETUR Compatibility**
Special output formats were added for full compatibility with TBG-ETUR’s tile-based upscaling workflow. This ensures stable, per-tile segmentation masks for high-resolution refinement.

-**Impact Pack SEGS Support**
The node now works directly with Impact Pack SEGS, making SAM3 usable in automated and multi-stage SEGS pipelines.

-**Unified Model folder with other SAM3 nodes**
model at models/sam3/sam3.pt

New Cleanup Tools

Min-Size Filter: Removes tiny or unwanted segments below a defined size.

Fill Holes: Automatically fills empty gaps inside segmented regions.

This update makes the TBG-SAM3 node set fully usable inside ComfyUI, adding better compatibility, improved segmentation handling, and practical cleanup features for all of us.

## Features

- **SAM3 Model Loader and Downloader**: Simple loader for local, cached models plus an advanced loader that can auto-download sam3.pt from Hugging Face into models/sam3 and reuse it across sessions.
- **SAM3 Instance Filtering Switch**: Optional instances toggle to either return all SAM3 detections or keep only detections that overlap positive boxes / contain positive points, giving clean “instance‑aware” outputs when needed.
- **Min-Size Filter**: Removes segments smaller than a specified size to clean up the final result.
- **Fill Holes**: Automatically fills in empty regions within segmented areas.
- **Unified Prompt Segmentation**: Segmentation driven by text, interactive points, and boxes via a single prompt pipeline 
- **Text-Prompt Segmentation**: Semantic segmentation using flexible open vocabulary text prompts.
- **Point, Box & Mask Guidance**: Support for positive/negative points, positive/negative boxes, and optional input masks to refine or restrict SAM3 predictions.
- **TBG‑ETUR Upscaler Compatibility**: SEGS format and shapes are aligned with TBG‑ETUR’s tiler/upscaler (no zero‑sized masks, consistent SEG.cropped_mask and SEG.crop_region), so SAM3 segments can be passed straight into the Upscaler/Refiner workflows. [ComfyUI-TBG-ETUR](https://github.com/Ltamann/ComfyUI-TBG-ETUR)
- **Impact Pack Compatible SEGS Outputs**: Per-instance SEGS built via mask_to_segs (one SEG per contour / detection). Combined full-image mask plus combined SEGS using the same Impact-Pack SEG structure for seamless use with detailers and other SEGS nodes.  [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) 
- **Configurable Crop Factor for SEGS**: User-controlled crop_factor input to tune how tightly Impact-Pack SEG crop regions wrap around their masks, matching Impact Pack behavior. 
- **Flexible Pipeline Modes**: Switch between all prompts, boxes-only, points-only, positive-only, negative-only (and variants) to control how selectors influence segmentation.
- **Interactive Prompt Collector Node**: Web‑UI node that displays the image, lets you draw points and boxes, and normalizes them into a reusable SAM3 prompt pipeline.
- **CUDA and CPU Support**: Efficient usage of available GPU or fallback to CPU.
- **Automatic Dependency Management**: Installs all necessary Python packages and handles Python 3.13+ specific issues.
- **Hugging Face Auth Friendly**: Integrated guidance and automated support for model access token handling.


## Installation

1. Clone or copy this repository into your ComfyUI `custom_nodes` directory:


   git clone https://github.com/your-username/ComfyUI-TBG-SAM3.git


2. Change directory and install required Python dependencies:

   cd ComfyUI-TBG-SAM3
   pip install -r requirements.txt


## Hugging Face Model Access Tutorial

### : Request Access

The SAM3 model checkpoint and API access is hosted on Hugging Face under gated access by Meta AI. To use the model, you need approval:

- Visit [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
- Click **Request Access** and fill in the required info.
- Wait for your access to be approved (up to 24 hours).

###: Model Download

The “TBG SAM3 Model Loader and Downloader” node downloads or copies the official SAM3 checkpoint file sam3.pt directly into the models/sam3 folder inside your ComfyUI installation (e.g. ComfyUI/models/sam3/sam3.pt).

To use the downloader, you must authenticate with Hugging Face so the node can download `sam3.pt` from the model hub.

1. Create a read-access token in your Hugging Face account.  
2. Set an environment variable named `HF_TOKEN` to that token before starting ComfyUI, for example:  
   - Linux/macOS: `export HF_TOKEN="your_token"`  
   - Windows (cmd): `set HF_TOKEN=your_token`  
3. Start ComfyUI in that environment. The “TBG SAM3 Model Loader and Downloader” node will automatically use `HF_TOKEN` to fetch `sam3.pt` from Hugging Face into the `models/sam3` folder.

Alternative just copy the model sam3.pt into the models/sam3 folder inside your ComfyUI installation (e.g. ComfyUI/models/sam3/sam3.pt).

## Credits

- Meta AI for the SAM3 Model ([GitHub](https://github.com/facebookresearch/sam3))
- ComfyUI community for custom node integration support
- Hugging Face for hosting models and hub services

### Enjoy segmenting everything! 

Feel free to [open issues](https://github.com/your-username/ComfyUI-TBG-SAM3/issues) or contribute improvements.

