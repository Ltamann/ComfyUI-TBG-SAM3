"""
ComfyUI SAM3 Node Package

TBG SAM3 integration for ComfyUI.
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

# Mandatory exports
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

installation_ok = False

# Optional: check whether the sam3 python package is present
try:
    import sam3  # noqa: F401
    print("[SAM3] SAM3 Python package detected.")
    installation_ok = True
except ImportError as e:
    print(f"[SAM3] SAM3 Python package not found or incomplete: {e}")
    print("[SAM3] NOTE: Local sam3.pt loaders may still work; tracker/video may not.")
    print("[SAM3] To reinstall/repair, run install.py manually in this folder:")
    print("       python.exe ComfyUI\\custom_nodes\\tbg-sam3\\install.py")

# Load node mappings from nodes.py
try:
    from .nodes import NODE_CLASS_MAPPINGS as NODES
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as NAMES

    NODE_CLASS_MAPPINGS.update(NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(NAMES)

    print(f"[SAM3] Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
    for node_id in NODE_CLASS_MAPPINGS.keys():
        display = NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
        print(f"[SAM3]  - {display}")

    if not installation_ok:
        print("\n[SAM3] WARNING: SAM3 package not fully installed; some features may fail.")
    print("=" * 70)

except ImportError as e:
    print(f"[SAM3] Cannot load SAM3 nodes: {e}")
    print("[SAM3] SAM3 nodes will be unavailable until this is resolved.")
    print("=" * 70)
except Exception as e:
    import traceback
    print(f"[SAM3] Error while loading SAM3 nodes: {e}")
    traceback.print_exc()
    print("=" * 70)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
