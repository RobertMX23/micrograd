import os
from pathlib import Path

def find_and_add_cuda_to_path():
    """Automatically find CUDA installation and add to PATH if not present"""
    # Check if CUDA is already in PATH
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    if any("CUDA" in d and "bin" in d for d in path_dirs):
        return True  # Already in PATH
    
    # Try to find CUDA installation
    possible_paths = [
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
        Path("C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA"),
    ]
    
    for base_path in possible_paths:
        if base_path.exists():
            versions = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('v')]
            if versions:
                # Use the latest version
                latest = sorted(versions, key=lambda x: x.name, reverse=True)[0]
                cuda_bin = latest / "bin"
                if cuda_bin.exists():
                    # Add to current session PATH
                    current_path = os.environ.get("PATH", "")
                    cuda_bin_str = str(cuda_bin)
                    if cuda_bin_str not in current_path:
                        os.environ["PATH"] = cuda_bin_str + os.pathsep + current_path
                        return True
    return False

# Try to fix CUDA PATH automatically
find_and_add_cuda_to_path()

try:
    import cupy as xp
    GPU_AVAILABLE = True

    def sync():
        try:
            xp.cuda.stream.get_current_stream().synchronize()
        except Exception:
            pass
except Exception:
    import numpy as xp
    GPU_AVAILABLE = False

    def sync():
        return

