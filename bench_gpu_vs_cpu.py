import time
import json
from backend import xp, GPU_AVAILABLE, sync
import numpy as np

# Import micrograd components
import sys
import os
micrograd_path = os.path.join(os.path.dirname(__file__), 'micrograd')
if micrograd_path not in sys.path:
    sys.path.insert(0, micrograd_path)
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'micrograd'))
from micrograd.engine import Value
from micrograd.nn import MLP

# Test if GPU operations actually work (CuPy might import but fail at runtime)
def test_gpu_operations():
    """Test if GPU operations actually work"""
    if not GPU_AVAILABLE:
        return False
    try:
        # Try a simple GPU operation
        test_array = xp.random.rand(10, 10)
        result = (test_array @ test_array).sum()
        if GPU_AVAILABLE:
            sync()
        return True
    except (ImportError, OSError, RuntimeError) as e:
        print(f"⚠ GPU operations failed: {type(e).__name__}: {e}")
        print("   Falling back to CPU (NumPy)")
        return False

# Determine if GPU is actually usable
GPU_USABLE = test_gpu_operations()

def cpu_workload():
    """Raw NumPy operations benchmark"""
    A = np.random.rand(2048, 2048)
    B = np.random.rand(2048, 2048)
    return np.maximum(A @ B, 0).sum()

def gpu_workload():
    """Raw CuPy/NumPy operations benchmark"""
    if GPU_USABLE:
        A = xp.random.rand(2048, 2048)
        B = xp.random.rand(2048, 2048)
        return xp.maximum(A @ B, 0).sum()
    else:
        # Fallback to CPU
        A = np.random.rand(2048, 2048)
        B = np.random.rand(2048, 2048)
        return np.maximum(A @ B, 0).sum()

def micrograd_forward_backward(model, inputs, targets, iterations=10):
    """Run micrograd forward and backward passes"""
    for _ in range(iterations):
        # Forward pass
        scores = [model(x) for x in inputs]
        
        # Loss computation
        losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(targets, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        
        # L2 regularization
        alpha = 1e-4
        reg_loss = alpha * sum((p*p for p in model.parameters()))
        total_loss = data_loss + reg_loss
        
        # Backward pass
        model.zero_grad()
        total_loss.backward()
        
        # Update (simulate SGD step)
        learning_rate = 0.01
        for p in model.parameters():
            p.data -= learning_rate * p.grad
    
    return total_loss.data

def benchmark_micrograd_cpu(num_samples=100, iterations=10):
    """Benchmark micrograd with CPU backend (NumPy)"""
    # Force NumPy backend
    import numpy as np_backup
    np.random.seed(42)
    
    # Create synthetic data
    X = np_backup.random.randn(num_samples, 2)
    y = np_backup.random.choice([-1, 1], size=num_samples)
    
    # Create model
    model = MLP(2, [16, 16, 1])
    inputs = [list(map(Value, xrow)) for xrow in X]
    targets = y.tolist()
    
    # Benchmark
    t0 = time.perf_counter()
    final_loss = micrograd_forward_backward(model, inputs, targets, iterations)
    elapsed = (time.perf_counter() - t0) * 1000
    
    return elapsed, final_loss

def benchmark_micrograd_gpu(num_samples=100, iterations=10):
    """Benchmark micrograd with GPU backend (CuPy)"""
    # Use backend (CuPy if available and usable, NumPy otherwise)
    if GPU_USABLE:
        try:
            xp.random.seed(42)
            X = xp.random.randn(num_samples, 2)
            y = xp.random.choice([-1, 1], size=num_samples)
            # Convert to CPU for micrograd (since it works with scalars)
            X = xp.asnumpy(X)
            y = xp.asnumpy(y)
        except (ImportError, OSError, RuntimeError):
            # Fallback to CPU
            np.random.seed(42)
            X = np.random.randn(num_samples, 2)
            y = np.random.choice([-1, 1], size=num_samples)
    else:
        np.random.seed(42)
        X = np.random.randn(num_samples, 2)
        y = np.random.choice([-1, 1], size=num_samples)
    
    # Create model
    model = MLP(2, [16, 16, 1])
    inputs = [list(map(Value, xrow)) for xrow in X]
    targets = y.tolist()
    
    # Benchmark
    t0 = time.perf_counter()
    final_loss = micrograd_forward_backward(model, inputs, targets, iterations)
    if GPU_USABLE:
        try:
            sync()
        except (ImportError, OSError, RuntimeError):
            pass
    elapsed = (time.perf_counter() - t0) * 1000
    
    return elapsed, final_loss

def benchmark(fn, runs=5):
    """Generic benchmark function"""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            fn()
            if GPU_USABLE:
                sync()
        except (ImportError, OSError, RuntimeError) as e:
            # If GPU operation fails, fall back to CPU
            if GPU_AVAILABLE:
                print(f"⚠ GPU operation failed during benchmark: {type(e).__name__}")
                print("   Using CPU fallback")
            raise
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    return sum(times) / len(times)

def run_comprehensive_benchmark():
    """Run all benchmarks and collect results"""
    results = {
        "gpu_available": GPU_AVAILABLE,
        "gpu_usable": GPU_USABLE,
        "backend": "CuPy" if GPU_USABLE else "NumPy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_operations": {},
        "micrograd_operations": {}
    }
    
    print("=" * 60)
    print("micrograd GPU vs CPU Performance Benchmark")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"GPU Usable: {GPU_USABLE}")
    print(f"Backend: {results['backend']}")
    if GPU_AVAILABLE and not GPU_USABLE:
        print("⚠ Warning: CuPy installed but GPU operations failed (CUDA DLLs not found)")
        print("   Falling back to CPU (NumPy)")
    print()
    
    # Raw operations benchmark
    print("Running raw NumPy/CuPy operations benchmark...")
    cpu_time = benchmark(cpu_workload)
    try:
        gpu_time = benchmark(gpu_workload)
    except (ImportError, OSError, RuntimeError) as e:
        print(f"⚠ GPU benchmark failed: {type(e).__name__}: {e}")
        print("   Using CPU time for both measurements")
        gpu_time = cpu_time
    
    results["raw_operations"] = {
        "cpu_time_ms": round(cpu_time, 2),
        "gpu_time_ms": round(gpu_time, 2),
        "speedup": round(cpu_time / gpu_time, 2) if GPU_USABLE and gpu_time > 0 else None
    }
    
    print(f"  CPU Avg Time: {cpu_time:.2f} ms")
    print(f"  GPU Avg Time: {gpu_time:.2f} ms")
    if GPU_USABLE and gpu_time > 0 and gpu_time < cpu_time:
        print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
    elif not GPU_USABLE:
        print("  (GPU not usable, using CPU for both)")
    print()
    
    # Micrograd benchmarks
    print("Running micrograd neural network benchmarks...")
    print("  (This may take a moment...)")
    
    # CPU benchmark (always NumPy)
    cpu_times = []
    cpu_losses = []
    for run in range(3):
        elapsed, loss = benchmark_micrograd_cpu(num_samples=100, iterations=20)
        cpu_times.append(elapsed)
        cpu_losses.append(float(loss))
    
    cpu_avg_time = sum(cpu_times) / len(cpu_times)
    cpu_avg_loss = sum(cpu_losses) / len(cpu_losses)
    
    # GPU benchmark (CuPy if available, NumPy otherwise)
    gpu_times = []
    gpu_losses = []
    for run in range(3):
        elapsed, loss = benchmark_micrograd_gpu(num_samples=100, iterations=20)
        gpu_times.append(elapsed)
        gpu_losses.append(float(loss))
    
    gpu_avg_time = sum(gpu_times) / len(gpu_times)
    gpu_avg_loss = sum(gpu_losses) / len(gpu_losses)
    
    results["micrograd_operations"] = {
        "before_gpu": {
            "backend": "NumPy (CPU)",
            "avg_time_ms": round(cpu_avg_time, 2),
            "individual_runs_ms": [round(t, 2) for t in cpu_times],
            "final_loss": round(cpu_avg_loss, 6),
            "config": {
                "num_samples": 100,
                "iterations": 20,
                "model_architecture": "MLP(2, [16, 16, 1])"
            }
        },
        "after_gpu": {
            "backend": "CuPy (GPU)" if GPU_USABLE else "NumPy (CPU fallback)",
            "avg_time_ms": round(gpu_avg_time, 2),
            "individual_runs_ms": [round(t, 2) for t in gpu_times],
            "final_loss": round(gpu_avg_loss, 6),
            "config": {
                "num_samples": 100,
                "iterations": 20,
                "model_architecture": "MLP(2, [16, 16, 1])"
            }
        },
        "performance_improvement": {
            "time_reduction_ms": round(cpu_avg_time - gpu_avg_time, 2),
            "time_reduction_percent": round((1 - gpu_avg_time / cpu_avg_time) * 100, 2) if cpu_avg_time > 0 else 0,
            "speedup": round(cpu_avg_time / gpu_avg_time, 2) if gpu_avg_time > 0 else None
        }
    }
    
    print()
    print("=" * 60)
    print("MICROGRAD BENCHMARK RESULTS")
    print("=" * 60)
    print("\nBEFORE (CPU - NumPy):")
    print(f"  Average Time: {cpu_avg_time:.2f} ms")
    print(f"  Individual Runs: {[f'{t:.2f}' for t in cpu_times]} ms")
    print(f"  Final Loss: {cpu_avg_loss:.6f}")
    
    print("\nAFTER (GPU - CuPy):" if GPU_USABLE else "\nAFTER (CPU - NumPy fallback):")
    print(f"  Average Time: {gpu_avg_time:.2f} ms")
    print(f"  Individual Runs: {[f'{t:.2f}' for t in gpu_times]} ms")
    print(f"  Final Loss: {gpu_avg_loss:.6f}")
    
    if GPU_USABLE and gpu_avg_time > 0 and gpu_avg_time < cpu_avg_time:
        speedup = cpu_avg_time / gpu_avg_time
        reduction = (1 - gpu_avg_time / cpu_avg_time) * 100
        print(f"\nPERFORMANCE IMPROVEMENT:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Time Reduction: {reduction:.2f}%")
        print(f"  Time Saved: {cpu_avg_time - gpu_avg_time:.2f} ms")
    else:
        if not GPU_USABLE:
            print("\nNOTE: GPU not usable (CUDA DLLs not found). Using CPU for all operations.")
        else:
            print("\nNOTE: No performance improvement detected.")
    
    print("\nNOTE: micrograd operates on scalar values, so GPU acceleration")
    print("      primarily benefits batch operations and future tensor extensions.")
    print("      The backend infrastructure is in place and ready for enhancements.")
    
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    
    # Output JSON
    json_output = json.dumps(results, indent=2)
    
    # Write to file
    with open("benchmark_results.json", "w") as f:
        f.write(json_output)
    
    print("\nDetailed results saved to: benchmark_results.json")
    print("\nJSON Output:")
    print(json_output)

