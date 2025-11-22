import subprocess
import sys
import os

DEPENDENCIES = [
    "timm>=1.0.17",
    "numpy>=1.26",  # Will pick the newest that exists for your Python
    "tqdm",
    "ftfy>=6.1.1",
    "regex",
    "iopath>=0.1.10",
    "typing_extensions",
    # Match transformers requirement: >=0.34.0,<1.0
    "huggingface_hub>0.36.0",
    # Optional but recommended: keep transformers itself in a safe range
    "transformers>=4.40.0,<5.0.0",
]

def run(cmd, desc):
    print(f"[SAM3 Install] {desc} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[SAM3 Install] FAILED: {desc}\n{result.stderr}")
        sys.exit(1)
    print(f"[SAM3 Install] OK: {desc}")

def main():
    print("=" * 70)
    print("[SAM3 Install] ComfyUI SAM3 Node - Installation (Modern Robust Style)")
    print("=" * 70)
    # Step 1: Install dependencies, always >= or latest
    print("[SAM3 Install] Step 1: Installing dependencies (unpinned)...")
    for dep in DEPENDENCIES:
        run([sys.executable, "-m", "pip", "install", "--upgrade", dep], f"Install {dep}")

    # Step 2: Clone SAM3 if not present (optional, comment out if always cloned before)
    if not os.path.exists("sam3"):
        run(["git", "clone", "https://github.com/facebookresearch/sam3.git"], "Clone SAM3 repo")
    else:
        print("[SAM3 Install] SAM3 directory exists, skipping clone.")

    # Step 3: Install SAM3 in editable mode, no dependency check (immune to repo pinning)
    run([sys.executable, "-m", "pip", "install", "-e", "./sam3", "--no-deps"], "Install SAM3 (editable, no deps)")

    print("\n[SAM3 Install] ✓ All dependencies and SAM3 installed successfully!")
    print("[SAM3 Install] If you see errors, check PyTorch/numpy compatibility and try again.\n")

if __name__ == "__main__":
    main()
