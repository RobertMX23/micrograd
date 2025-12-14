import subprocess
import sys
import json
import os

def main():
    print("=== micrograd GPU Performance Demo ===")

    try:
        import backend
        if backend.GPU_AVAILABLE:
            print("[OK] GPU detected. Using hardware acceleration (CuPy).")
        else:
            print("[WARNING] GPU not detected. Falling back to CPU (NumPy).")
    except Exception as e:
        print("Backend error:", e)
        sys.exit(1)
    
    # Check if CUDA setup needs fixing
    print("\nChecking CUDA setup...")
    try:
        result = subprocess.run(
            [sys.executable, "check_cuda_setup.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Check if PATH issue is detected
        if "CUDA not in PATH" in result.stdout or "GPU operations failing" in result.stdout:
            print("\n[WARNING] CUDA PATH issue detected. Attempting to fix...")
            fix_result = subprocess.run(
                [sys.executable, "fix_cuda_path.py"],
                timeout=30
            )
            if fix_result.returncode == 0:
                print("[OK] CUDA PATH fixed! Restarting backend...")
                # Reload backend to pick up new PATH
                import importlib
                importlib.reload(backend)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Scripts not found or timeout - continue anyway
        pass
    except Exception as e:
        # Ignore errors in diagnostic scripts
        pass

    print("\nRunning comprehensive benchmark...\n")
    try:
        subprocess.run([sys.executable, "bench_gpu_vs_cpu.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[WARNING] Benchmark failed with exit code {e.returncode}")
        print("   Check the error messages above for details.")
        print("\n[TIP] Try running: python fix_cuda_path.py")
        sys.exit(1)

    # Display JSON results if available
    if os.path.exists("benchmark_results.json"):
        print("\n" + "=" * 60)
        print("Loading detailed JSON results...")
        print("=" * 60)
        with open("benchmark_results.json", "r") as f:
            results = json.load(f)
        
        print("\nSummary:")
        print(f"  Backend: {results['backend']}")
        print(f"  GPU Available: {results['gpu_available']}")
        
        if results.get('micrograd_operations'):
            mg = results['micrograd_operations']
            before = mg['before_gpu']
            after = mg['after_gpu']
            
            print(f"\nMicrograd Performance:")
            print(f"  Before (CPU): {before['avg_time_ms']} ms")
            print(f"  After ({after['backend']}): {after['avg_time_ms']} ms")
            
            if mg['performance_improvement']['speedup']:
                print(f"  Speedup: {mg['performance_improvement']['speedup']}x")
                print(f"  Time Reduction: {mg['performance_improvement']['time_reduction_percent']}%")

    print("\n[OK] Demo complete.")
    print(f"  Full results saved to: benchmark_results.json")

if __name__ == "__main__":
    main()

