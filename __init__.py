"""
ComfyUI SAM3 Node Package

TBG SAM3 integration for ComfyUI.
- Does NOT auto-run installers at import.
- Always tries to load nodes.py, even if SAM3 tracker pieces are missing.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print("\n" + "=" * 70)
print("[SAM3] ComfyUI SAM3 Node - Loading (TBG)")
print("=" * 70)

# Exports required by ComfyUI
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

installation_ok = False

# 1) Optional: check whether the SAM3 Python package is importable
try:
    import sam3  # noqa: F401
    print("[SAM3] SAM3 Python package detected.")
    installation_ok = True
except ImportError as e:
    print(f"[SAM3] SAM3 Python package not fully available: {e}")
    print("[SAM3] Image-only loaders that use local sam3.pt may still work.")
    print("[SAM3] To reinstall/repair SAM3, run install.py in this folder:")
    print("       python.exe ComfyUI\\custom_nodes\\tbg-sam3\\install.py")

# 2) Load node mappings from nodes.py
try:
    from .nodes import NODE_CLASS_MAPPINGS as NODES
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as NAMES

    NODE_CLASS_MAPPINGS.update(NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(NAMES)

    print(f"[SAM3] Registered {len(NODE_CLASS_MAPPINGS)} nodes:")
    for node_id, cls in NODE_CLASS_MAPPINGS.items():
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
        print(f"[SAM3]  - {display_name} ({node_id})")

    if not installation_ok:
        print("\n[SAM3] WARNING: SAM3 package not fully installed; tracker/video may be unavailable.")
    print("=" * 70)

except Exception as e:
    import traceback
    print(f"[SAM3] ERROR while loading SAM3 nodes: {e}")
    traceback.print_exc()
    print("[SAM3] SAM3 nodes will be unavailable until this is fixed.")
    print("=" * 70)

# Optional: web assets directory (for JS widgets)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
