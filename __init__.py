"""
ComfyUI SAM3 Node Package
Official SAM3 integration with auto-installation
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print("\n" + "=" * 70)
print("[SAM3] ComfyUI SAM3 Node - Loading")
print("=" * 70)

# Initialize empty mappings at module level - CRITICAL
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Run installation check
installation_ok = False

try:
    from .install import check_pytorch, install_requirements, install_sam3

    # Check PyTorch
    if not check_pytorch():
        print("[SAM3] WARNING: PyTorch check failed")

    # Check SAM3 installation
    try:
        import sam3
        print("[SAM3] SAM3 already installed")
        installation_ok = True
    except ImportError:
        print("[SAM3] SAM3 not found - attempting auto-installation...")
        install_requirements()
        if install_sam3():
            installation_ok = True
        else:
            print("[SAM3] Auto-installation failed")
            print("[SAM3] Manual install: pip install git+https://github.com/facebookresearch/sam3.git")

except Exception as e:
    print(f"[SAM3] Installation check error: {e}")

# Load nodes if installation successful
try:
    # Import the mappings from nodes.py
    from .nodes import NODE_CLASS_MAPPINGS as NODES
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as NAMES

    # Update module-level mappings
    NODE_CLASS_MAPPINGS.update(NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(NAMES)

    print(f"[SAM3] Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
    for node_id in NODE_CLASS_MAPPINGS.keys():
        display = NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
        print(f"[SAM3]   • {display}")

    if not installation_ok:
        print("\n[SAM3] IMPORTANT: HuggingFace auth required:")
        print("[SAM3]   1. Request access: https://huggingface.co/facebook/sam3")
        print("[SAM3]   2. Run: huggingface-cli login")
        print("[SAM3]   3. Restart ComfyUI")

    print("\n[SAM3] Model Loading Options:")
    print("[SAM3]   1. Auto-download: Select 'auto (API)' in SAM3ModelLoader")
    print("[SAM3]   2. Local models: Place in models/sam3/ folder")

except ImportError as e:
    print(f"[SAM3] Cannot load: {e}")
    print("[SAM3] SAM3 not installed")
except Exception as e:
    print(f"[SAM3] Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)

# CRITICAL: Export at module level
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
