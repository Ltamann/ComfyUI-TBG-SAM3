```markdown
# ComfyUI-TBG-SAM3

A ComfyUI custom node extension integrating Meta's **Segment Anything Model 3 (SAM 3)** for advanced image and video segmentation capabilities. This extension provides production-ready nodes compatible with ComfyUI’s Impact Pack SEGS format, enabling text-prompt, point-based, and mask-driven segmentation as well as depth map generation per segment or for full images.

Developed and tested for Python 3.13+ and ComfyUI 0.3.60 and above, with automatic handling of model downloading, dependency installation, and robust fallback for Python versions beyond SAM3’s original constraints.

---

## Features

- **SAM3 Model Loader Node**: Automatically downloads and loads the SAM3 model checkpoint with Hugging Face integration.
- **Text-Prompt Segmentation**: Semantic segmentation using flexible open vocabulary text prompts.
- **Point & Mask Guided Segmentation**: Select objects interactively by points or masks.
- **Impact Pack Compatible SEGS Output**: Full multi-instance segmentation output compatible with ComfyUI’s downstream nodes.
- **Depth Map Generation Node**: Generate depth maps for entire images or individual segments using MiDaS.
- **CUDA and CPU Support**: Efficient usage of available GPU or fallback to CPU.
- **Automatic Dependency Management**: Installs all necessary Python packages and handles Python 3.13+ specific issues.
- **Hugging Face Auth Friendly**: Integrated guidance and automated support for model access token handling.

---

## Installation

1. Clone or copy this repository into your ComfyUI `custom_nodes` directory:

   ```
   git clone https://github.com/your-username/ComfyUI-TBG-SAM3.git
   ```

2. Change directory and install required Python dependencies:

   ```
   cd ComfyUI-TBG-SAM3
   pip install -r requirements.txt
   ```

3. Install SAM3 package (Python 3.13+ compatible mode automatically handled):

   ```
   pip install git+https://github.com/facebookresearch/sam3.git --no-deps
   ```

4. Install additional dependencies required by SAM3 at runtime (including video decoding and dataset tooling):

   ```
   pip install decord av hydra-core omegaconf iopath einops tqdm pycocotools shapely
   ```

5. Restart ComfyUI.

---

## Hugging Face Model Access Tutorial

### Step 1: Request Access

The SAM3 model checkpoint is hosted on Hugging Face under gated access by Meta AI. To use the model, you need approval:

- Visit [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
- Click **Request Access** and fill in the required info.
- Wait for your access to be approved (up to 24 hours).

### Step 2: Authenticate Locally

Once your access is approved, authenticate using the Hugging Face CLI on your machine:

1. Open terminal or command prompt.
2. Run the login command:

   ```
   huggingface-cli login
   ```

3. Paste your Hugging Face token when prompted. You can generate a token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with read permissions.

### Step 3: Set Environment Variable (Optional)

Alternatively, you can set your token as an environment variable to avoid running login each time:

- **Windows CMD:**

  ```
  set HF_TOKEN=your_token_here
  ```

- **PowerShell:**

  ```
  $env:HF_TOKEN="your_token_here"
  ```

- Restart ComfyUI after setting this.

### Step 4: Model Download

The SAM3 Model Loader node in ComfyUI will automatically download the model checkpoint to `models/sam3/` on first run once you have authenticated.

---

## Usage Guide

1. Add and run the **SAM3 Model Loader** node. Choose your device (`cuda` recommended).

2. Pass images through the **SAM3 Segmentation** node:
   - Choose `mode = text` for text-based segmentation with your prompt.
   - Choose `mode = points_from_mask` to segment by interactive mask points.
   - Use `auto` mode for combined fallback.

3. Optional: Use the **SAM3 Depth Map** node to generate depth visuals for full images or individual segments.

---

## Troubleshooting

- **Model Download Failures**: Ensure Hugging Face access is approved and you have logged in or set the `HF_TOKEN`.
- **Missing Python Packages**: Use the installation commands above to install all dependencies.
- **Python Version Issues**: This extension supports Python 3.13+, with improvements to relax SAM3’s original numpy constraint.
- **Performance and Memory**: Use CUDA-enabled GPUs if available. Otherwise, expect slower CPU performance.

---

## Credits

- Meta AI for the SAM3 Model ([GitHub](https://github.com/facebookresearch/sam3))
- ComfyUI community for custom node integration support
- Hugging Face for hosting models and hub services

---

## License

MIT License — see [LICENSE](./LICENSE) for details.

---

### Enjoy segmenting everything! 🚀

Feel free to [open issues](https://github.com/your-username/ComfyUI-TBG-SAM3/issues) or contribute improvements.
```
