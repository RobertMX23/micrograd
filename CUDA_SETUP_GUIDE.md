# CUDA Setup Guide

## Quick Fix

If you're getting `nvrtc64_*.dll` or `DLL load failed` errors, run:

```powershell
python fix_cuda_path.py
```

Or double-click: `fix_cuda.bat`

## Diagnostic Tools

### Check CUDA Setup
```powershell
python check_cuda_setup.py
```
This will show you:
- CUDA installation status
- PATH configuration
- CuPy installation
- GPU operation status

### Fix CUDA PATH
```powershell
python fix_cuda_path.py
```
This script will:
1. Find your CUDA installation
2. Add CUDA bin directory to PATH
3. Test if GPU operations work
4. Provide troubleshooting steps if issues persist

## Common Issues

### Issue: "CuPy failed to load nvrtc64_120_0.dll"

**Cause**: CUDA version mismatch between installed CUDA and CuPy package.

**Solution**:
1. Check your CUDA version: `nvcc --version`
2. Reinstall CuPy matching your CUDA version:
   ```powershell
   pip uninstall cupy
   pip install cupy-cuda12x  # For CUDA 12.x
   # or
   pip install cupy-cuda11x  # For CUDA 11.x
   ```

### Issue: "CUDA not in PATH"

**Solution**: Run `python fix_cuda_path.py` - it will automatically add CUDA to PATH.

### Issue: "GPU Usable: False"

This means CuPy is installed but can't access CUDA DLLs. Usually caused by:
- CUDA not in PATH → Run `fix_cuda_path.py`
- Version mismatch → Reinstall matching CuPy version
- Incomplete CUDA installation → Reinstall CUDA Toolkit

## Automatic Fixes

The `backend.py` now automatically tries to find and add CUDA to PATH when imported.

The `run_gpu_demo.py` script will also attempt to fix PATH issues automatically before running benchmarks.

## Manual PATH Setup

If automatic fixes don't work:

1. Find your CUDA installation (usually):
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`

2. Add to System PATH:
   - Win + X → System → Advanced system settings
   - Environment Variables → System variables → Path → Edit
   - Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
   - Replace `v12.x` with your actual version

3. Restart PowerShell/terminal

## Verification

After setup, verify everything works:

```powershell
python check_cuda_setup.py
python run_gpu_demo.py
```

You should see `GPU Usable: True` and successful GPU benchmarks.
