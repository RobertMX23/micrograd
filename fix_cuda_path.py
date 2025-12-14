"""
CUDA PATH Fix Script
Automatically finds CUDA installation and adds it to PATH
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def find_cuda_installation():
    """Find CUDA installation directory"""
    possible_paths = [
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
        Path("C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA"),
    ]
    
    for base_path in possible_paths:
        if base_path.exists():
            # Find all CUDA versions
            versions = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('v')]
            if versions:
                # Return the latest version
                latest = sorted(versions, key=lambda x: x.name, reverse=True)[0]
                return latest
    
    return None

def get_cuda_bin_path(cuda_path):
    """Get the bin directory path"""
    return cuda_path / "bin"

def check_dll_exists(cuda_bin_path, dll_name="nvrtc64_120_0.dll"):
    """Check if a specific DLL exists in CUDA bin"""
    dll_path = cuda_bin_path / dll_name
    return dll_path.exists()

def add_to_path_permanently(cuda_bin_path):
    """Add CUDA bin to system PATH permanently (requires admin)"""
    import winreg
    
    try:
        # Open the system environment variables registry key
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            0,
            winreg.KEY_ALL_ACCESS
        )
        
        # Get current PATH
        current_path, _ = winreg.QueryValueEx(key, "Path")
        
        # Check if already in PATH
        path_str = str(cuda_bin_path)
        if path_str not in current_path:
            # Add to PATH
            new_path = current_path + os.pathsep + path_str
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
            print(f"[OK] Added to system PATH: {path_str}")
            
            # Broadcast environment change
            import ctypes
            ctypes.windll.user32.SendMessageW(0xFFFF, 0x001A, 0, "Environment")
        else:
            print(f"[OK] Already in PATH: {path_str}")
        
        winreg.CloseKey(key)
        return True
    except PermissionError:
        print("[WARNING] Admin rights required for permanent PATH modification")
        return False
    except Exception as e:
        print(f"[WARNING] Error modifying PATH: {e}")
        return False

def add_to_path_temporary(cuda_bin_path):
    """Add CUDA bin to PATH for current session"""
    path_str = str(cuda_bin_path)
    current_path = os.environ.get("PATH", "")
    
    if path_str not in current_path:
        os.environ["PATH"] = path_str + os.pathsep + current_path
        print(f"[OK] Added to current session PATH: {path_str}")
    else:
        print(f"[OK] Already in current session PATH: {path_str}")

def get_cuda_version_from_path(cuda_path):
    """Extract CUDA version from path (e.g., v12.0 -> 12.0)"""
    version_str = cuda_path.name.lstrip('v')
    # Extract major.minor
    parts = version_str.split('.')
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version_str

def get_cupy_version_info():
    """Get CuPy version and what CUDA version it expects"""
    try:
        import cupy as cp
        version = cp.__version__
        # Try to determine expected CUDA version from error messages or package name
        return True, version, None
    except ImportError:
        return False, None, None
    except Exception as e:
        return None, str(e), None

def test_cuda_import():
    """Test if CuPy can now import CUDA libraries"""
    try:
        import cupy as cp
        # Try a simple operation that requires CUDA DLLs
        test_array = cp.random.rand(10, 10)
        result = (test_array @ test_array).sum()
        cp.cuda.stream.get_current_stream().synchronize()
        print("[OK] CUDA operations working!")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] CUDA operations still failing: {type(e).__name__}: {error_msg}")
        
        # Check for version mismatch
        if "nvrtc64_" in error_msg or "dll" in error_msg.lower():
            print("\n[DIAGNOSIS] This looks like a CUDA version mismatch.")
            print("   CuPy may be compiled for a different CUDA version than installed.")
            print("   Solution: Reinstall CuPy matching your CUDA version:")
            print("     pip uninstall cupy cupy-cuda12x cupy-cuda11x cupy-cuda13x")
            # Try to detect CUDA version and suggest correct package
            try:
                result = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "release 13" in result.stdout:
                    print("     pip install cupy-cuda13x  # For CUDA 13.x")
                elif "release 12" in result.stdout:
                    print("     pip install cupy-cuda12x  # For CUDA 12.x")
                elif "release 11" in result.stdout:
                    print("     pip install cupy-cuda11x  # For CUDA 11.x")
                else:
                    print("     pip install cupy-cuda12x  # or cupy-cuda11x, cupy-cuda13x, etc.")
            except:
                print("     pip install cupy-cuda12x  # or cupy-cuda11x, cupy-cuda13x, etc.")
        
        return False

def main():
    print("=" * 60)
    print("CUDA PATH Fix Script")
    print("=" * 60)
    print()
    
    # Find CUDA installation
    print("Searching for CUDA installation...")
    cuda_path = find_cuda_installation()
    
    if not cuda_path:
        print("[ERROR] CUDA installation not found!")
        print("\nPlease ensure CUDA Toolkit is installed from:")
        print("  https://developer.nvidia.com/cuda-downloads")
        return 1
    
    print(f"[OK] Found CUDA at: {cuda_path}")
    
    # Get CUDA version
    cuda_version = get_cuda_version_from_path(cuda_path)
    print(f"[OK] CUDA version: {cuda_version}")
    
    # Get bin directory
    cuda_bin = get_cuda_bin_path(cuda_path)
    print(f"[OK] CUDA bin directory: {cuda_bin}")
    
    # Check if bin directory exists
    if not cuda_bin.exists():
        print(f"[ERROR] CUDA bin directory not found: {cuda_bin}")
        return 1
    
    # Check for DLLs
    print("\nChecking for CUDA DLLs...")
    # Check for any nvrtc DLLs (runtime compilation)
    nvrtc_dlls = list(cuda_bin.glob("nvrtc64_*.dll"))
    # Check for cudart DLLs (CUDA runtime)
    cudart_dlls = list(cuda_bin.glob("cudart64_*.dll"))
    # Check for all DLLs
    all_dlls = list(cuda_bin.glob("*.dll"))
    
    found_dlls = nvrtc_dlls + cudart_dlls
    
    if found_dlls:
        print(f"[OK] Found {len(found_dlls)} CUDA DLL(s) (NVRTC/CUDART)")
        if nvrtc_dlls:
            print(f"   NVRTC DLLs: {[d.name for d in nvrtc_dlls[:3]]}")
        if cudart_dlls:
            print(f"   CUDA Runtime DLLs: {[d.name for d in cudart_dlls[:3]]}")
    elif all_dlls:
        print(f"[INFO] Found {len(all_dlls)} DLL(s) in bin directory (but no NVRTC/CUDART)")
        print(f"   Sample DLLs: {[d.name for d in all_dlls[:5]]}")
    else:
        print("[WARNING] No DLLs found in bin directory")
        print("   Checking alternative locations...")
        # Check lib directory
        lib_path = cuda_path / "lib" / "x64"
        if lib_path.exists():
            lib_dlls = list(lib_path.glob("*.dll"))
            if lib_dlls:
                print(f"   Found {len(lib_dlls)} DLL(s) in lib/x64 directory")
                nvrtc_lib = list(lib_path.glob("nvrtc64_*.dll"))
                if nvrtc_lib:
                    print(f"   NVRTC DLLs in lib: {[d.name for d in nvrtc_lib[:3]]}")
                    # Suggest adding lib to PATH too
                    print("   [NOTE] You may need to add lib/x64 to PATH as well")
    
    # Check CuPy version
    print("\nChecking CuPy installation...")
    cupy_ok, cupy_version, _ = get_cupy_version_info()
    if cupy_ok:
        print(f"[OK] CuPy installed: version {cupy_version}")
        # Determine expected CUDA version from CuPy package
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "cupy"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "cupy-cuda" in result.stdout.lower():
                print("[INFO] CuPy CUDA-specific package detected")
        except:
            pass
    elif cupy_ok is False:
        print("[ERROR] CuPy not installed")
    else:
        print(f"[WARNING] CuPy import error: {cupy_version}")
    
    # Add to PATH
    print("\n" + "=" * 60)
    print("Fixing PATH...")
    print("=" * 60)
    
    # Try permanent first (requires admin)
    permanent_success = add_to_path_permanently(cuda_bin)
    
    # Always add to current session
    add_to_path_temporary(cuda_bin)
    
    # Test if it works
    print("\n" + "=" * 60)
    print("Testing CUDA...")
    print("=" * 60)
    
    if test_cuda_import():
        print("\n[SUCCESS] CUDA is now working!")
        if not permanent_success:
            print("\n[NOTE] PATH was only added for this session.")
            print("   Restart your terminal/PowerShell for permanent changes.")
            print("   Or run this script as Administrator for permanent PATH modification.")
        return 0
    else:
        print("\n[WARNING] CUDA still not working.")
        print("\nTroubleshooting steps:")
        print("   1. Restart your terminal/PowerShell (to reload PATH)")
        print("   2. Check CuPy version matches CUDA version:")
        print(f"      - Installed CUDA: {cuda_version}")
        print("      - Reinstall matching CuPy:")
        major_version = cuda_version.split('.')[0]
        print(f"        pip uninstall cupy cupy-cuda12x cupy-cuda11x cupy-cuda13x")
        print(f"        pip install cupy-cuda{major_version}x")
        print("   3. Run this script as Administrator for permanent PATH changes")
        print("   4. Verify CUDA installation is complete")
        return 1

if __name__ == "__main__":
    sys.exit(main())
