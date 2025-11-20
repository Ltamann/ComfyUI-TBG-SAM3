**This is just the first version, so please be patient—there’s still a lot to fix and improve before it’s production-ready.**

# ComfyUI-TBG-SAM3

A ComfyUI custom node extension integrating Meta's **Segment Anything Model 3 (SAM 3)** for advanced image and video segmentation capabilities. This extension provides production-ready nodes compatible with ComfyUI’s Impact Pack SEGS format, enabling text-prompt, point-based, and mask-driven segmentation as well as depth map generation per segment or for full images.

Developed and tested for Python 3.13+ and ComfyUI 0.3.60 and above, with automatic handling of model downloading, dependency installation, and robust fallback for Python versions beyond SAM3’s original constraints.

## Features

- **SAM3 Model Loader Node**: Automatically downloads and loads the SAM3 model checkpoint with Hugging Face integration.
- **Text-Prompt Segmentation**: Semantic segmentation using flexible open vocabulary text prompts.
- **Point & Mask Guided Segmentation**: Select objects interactively by points or masks.
- **Impact Pack Compatible SEGS Output**: Full multi-instance segmentation output compatible with ComfyUI’s downstream nodes.
- **Depth Map Generation Node**: Generate depth maps for entire images or individual segments using MiDaS.
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

The SAM3 Model Loader node in ComfyUI will automatically download the model checkpoint to into cache using API on first run once you have authenticated.


## Credits

- Meta AI for the SAM3 Model ([GitHub](https://github.com/facebookresearch/sam3))
- ComfyUI community for custom node integration support
- Hugging Face for hosting models and hub services

### Enjoy segmenting everything! 

Feel free to [open issues](https://github.com/your-username/ComfyUI-TBG-SAM3/issues) or contribute improvements.

