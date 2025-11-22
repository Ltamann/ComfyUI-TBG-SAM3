**This is was a the first version for testing .... get this version for now https://github.com/PozzettiAndrea/ComfyUI-SAM3**

# ComfyUI-TBG-SAM3

A ComfyUI custom node extension integrating Meta's **Segment Anything Model 3 (SAM 3)** for advanced image and video segmentation capabilities. This extension provides production-ready nodes compatible with ComfyUI’s Impact Pack SEGS format, enabling text-prompt, point-based, and mask-driven segmentation as well as depth map generation per segment or for full images.
A ComfyUI custom-node extension that integrates Meta’s **Segment Anything Model 3 (SAM-3)** for advanced image segmentation. It supports unified point and box selection, text-prompt segmentation, point-guided masks, and mask-driven refinement. Features include an **Instant Instance Selection toggle**, **compatibility with tile-based upscalers such as TBG-ETUR**, and full support for **Impact Pack SEGS formats**.

The extension provides both combined and separated mask outputs, offers production-ready ComfyUI nodes, includes optional depth-map generation, and is fully compatible with Python 3.13+ on both CUDA and CPU environments.

## Features

- **SAM3 Model Loader and Downloader**: Simple loader for local, cached models plus an advanced loader that can auto-download sam3.pt from Hugging Face into models/sam3 and reuse it across sessions.
- **SAM3 Instance Filtering Switch**: Optional instances toggle to either return all SAM3 detections or keep only detections that overlap positive boxes / contain positive points, giving clean “instance‑aware” outputs when needed.
- **Unified Prompt Segmentation**: Segmentation driven by text, interactive points, and boxes via a single prompt pipeline 
- **Text-Prompt Segmentation**: Semantic segmentation using flexible open vocabulary text prompts.
- **Point, Box & Mask Guidance**: Support for positive/negative points, positive/negative boxes, and optional input masks to refine or restrict SAM3 predictions.
- **TBG‑ETUR Upscaler Compatibility**: SEGS format and shapes are aligned with TBG‑ETUR’s tiler/upscaler (no zero‑sized masks, consistent SEG.cropped_mask and SEG.crop_region), so SAM3 segments can be passed straight into the Upscaler/Refiner workflows.
- **Impact Pack Compatible SEGS Outputs**: Per-instance SEGS built via mask_to_segs (one SEG per contour / detection). Combined full-image mask plus combined SEGS using the same Impact-Pack SEG structure for seamless use with detailers and other SEGS nodes.
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

