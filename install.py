import subprocess
import sys
import os



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


    # Step 1: Clone SAM3 if not present
    if not os.path.exists("sam3"):
        run(["git", "clone", "https://github.com/facebookresearch/sam3.git"], "Clone SAM3 repo")
    else:
        print("[SAM3 Install] SAM3 directory exists, skipping clone.")

    # Step 3: Install SAM3 in editable mode, no dependency check
    run(
        [sys.executable, "-m", "pip", "install", "-e", "./sam3", "--no-deps"],
        "Install SAM3 (editable, no deps)",
    )

    # Use plain ASCII to avoid UnicodeEncodeError on Windows consoles
    print("[SAM3 Install] If you see errors, check PyTorch/numpy compatibility and try again.\n")


if __name__ == "__main__":
    main()
