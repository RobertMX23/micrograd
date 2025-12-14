"""
CUDA Setup Checker
Diagnostic script to check CUDA and CuPy setup
"""
import os
import sys
from pathlib import Path

def check_cuda_path():
    """Check if CUDA is in PATH"""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    cuda_dirs = [d for d in path_dirs if "CUDA" in d and "bin" in d]
    return cuda_dirs

def check_cuda_installation():
    """Check for CUDA installation"""
    possible_paths = [
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
        Path("C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA"),
    ]
    
    found = []
    for base_path in possible_paths:
        if base_path.exists():
            versions = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.startswith('v')]
            if versions:
                found.append((base_path, versions))
    
    return found

def check_cupy():
    """Check CuPy installation"""
    try:
        import cupy as cp
        version = cp.__version__
        cuda_version = None
        try:
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
        except:
            pass
        return True, version, cuda_version
    except ImportError:
        return False, None, None
    except Exception as e:
        return None, str(e), None

def check_cuda_dlls():
    """Check if CUDA DLLs are accessible"""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    dlls_found = []
    
    for path_dir in path_dirs:
        if "CUDA" in path_dir and "bin" in path_dir:
            bin_path = Path(path_dir)
            if bin_path.exists():
                # Check for common CUDA DLLs
                dlls = list(bin_path.glob("*.dll"))
                if dlls:
                    dlls_found.extend([(bin_path, dll.name) for dll in dlls[:5]])  # First 5
    
    return dlls_found

def test_gpu_operations():
    """Test if GPU operations actually work"""
    try:
        import cupy as cp
        test_array = cp.random.rand(10, 10)
        result = (test_array @ test_array).sum()
        cp.cuda.stream.get_current_stream().synchronize()
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("CUDA Setup Diagnostic")
    print("=" * 60)
    print()
    
    # Check CUDA installation
    print("1. Checking CUDA Installation...")
    cuda_installs = check_cuda_installation()
    if cuda_installs:
        for base_path, versions in cuda_installs:
            print(f"   [OK] Found: {base_path}")
            print(f"     Versions: {', '.join(versions)}")
    else:
        print("   [ERROR] CUDA Toolkit not found")
    print()
    
    # Check PATH
    print("2. Checking PATH...")
    cuda_paths = check_cuda_path()
    if cuda_paths:
        print(f"   [OK] CUDA in PATH ({len(cuda_paths)} location(s)):")
        for path in cuda_paths:
            print(f"     - {path}")
    else:
        print("   [ERROR] CUDA not in PATH")
        print("     Run fix_cuda_path.py to fix this")
    print()
    
    # Check DLLs
    print("3. Checking CUDA DLLs...")
    dlls = check_cuda_dlls()
    if dlls:
        print(f"   [OK] Found {len(dlls)} DLL(s) in PATH:")
        for bin_path, dll_name in dlls[:5]:
            print(f"     - {dll_name} in {bin_path}")
    else:
        print("   [ERROR] No CUDA DLLs found in PATH")
    print()
    
    # Check CuPy
    print("4. Checking CuPy...")
    cupy_status, cupy_version, cuda_version = check_cupy()
    if cupy_status is True:
        print(f"   [OK] CuPy installed: version {cupy_version}")
        if cuda_version:
            print(f"   [OK] CUDA runtime version: {cuda_version}")
    elif cupy_status is False:
        print("   [ERROR] CuPy not installed")
        print("     Install with: pip install cupy-cuda12x")
    else:
        print(f"   [WARNING] CuPy import error: {cupy_version}")
    print()
    
    # Test GPU operations
    print("5. Testing GPU Operations...")
    if cupy_status is True:
        works, error = test_gpu_operations()
        if works:
            print("   [OK] GPU operations working!")
        else:
            print(f"   [ERROR] GPU operations failed: {error}")
            print("     Run fix_cuda_path.py to fix this")
    else:
        print("   [WARNING] Skipping (CuPy not available)")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    issues = []
    if not cuda_installs:
        issues.append("CUDA Toolkit not installed")
    if not cuda_paths:
        issues.append("CUDA not in PATH")
    if cupy_status is False:
        issues.append("CuPy not installed")
    elif cupy_status is True:
        works, _ = test_gpu_operations()
        if not works:
            issues.append("GPU operations failing")
    
    if not issues:
        print("[OK] All checks passed! CUDA should be working.")
    else:
        print("[WARNING] Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nRecommended actions:")
        if "PATH" in str(issues):
            print("   1. Run: python fix_cuda_path.py")
        if "CuPy" in str(issues):
            print("   2. Install CuPy: pip install cupy-cuda12x")
        if "installed" in str(issues):
            print("   3. Install CUDA Toolkit from nvidia.com")

if __name__ == "__main__":
    main()
