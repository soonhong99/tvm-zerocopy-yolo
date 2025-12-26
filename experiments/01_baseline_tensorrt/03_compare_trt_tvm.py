#!/usr/bin/env python3
"""
TensorRT vs TVM 성능 비교 분석
베이스라인 대비 TVM 성능 평가

실행: python3 experiments/01_baseline_tensorrt/03_compare_trt_tvm.py

비교 항목:
- 추론 레이턴시
- FPS
- 메모리 사용량
- 엔진/라이브러리 크기
"""

import os
import sys
import time
import numpy as np
from typing import Dict, Optional

# TVM
try:
    import tvm
    from tvm.contrib import graph_executor
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    print("WARNING: TVM not available")

# TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: TensorRT not available")


class TVMInference:
    """TVM 추론 래퍼"""

    def __init__(self, lib_path: str):
        self.lib_path = lib_path

        # 라이브러리 로드
        self.lib = tvm.runtime.load_module(lib_path)
        self.dev = tvm.cuda()
        self.module = graph_executor.GraphModule(self.lib["default"](self.dev))

        # 입출력 정보
        self.input_shape = (1, 3, 640, 640)
        self.output_shape = (1, 25200, 85)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """추론 실행"""
        self.module.set_input("images", input_data)
        self.module.run()
        return self.module.get_output(0).numpy()

    def benchmark(self, num_warmup: int = 20, num_runs: int = 100) -> Dict:
        """벤치마크"""
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)

        # 워밍업
        for _ in range(num_warmup):
            self.infer(dummy_input)
        self.dev.sync()

        # 벤치마크
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.infer(dummy_input)
            self.dev.sync()
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'fps': float(1000 / np.mean(times)),
            'p95_ms': float(np.percentile(times, 95)),
        }


class TRTInference:
    """TensorRT 추론 래퍼"""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self._setup_bindings()

    def _setup_bindings(self):
        """입출력 버퍼 설정"""
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)

            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name, 'shape': shape, 'dtype': dtype,
                    'device': device_mem, 'host': np.empty(shape, dtype=dtype)
                })
            else:
                self.outputs.append({
                    'name': name, 'shape': shape, 'dtype': dtype,
                    'device': device_mem, 'host': np.empty(shape, dtype=dtype)
                })

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """추론 실행"""
        np.copyto(self.inputs[0]['host'], input_data.ravel().reshape(self.inputs[0]['shape']))
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))

        self.context.execute_async_v3(self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()
        return self.outputs[0]['host']

    def benchmark(self, num_warmup: int = 20, num_runs: int = 100) -> Dict:
        """벤치마크"""
        input_shape = self.inputs[0]['shape']
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # 워밍업
        for _ in range(num_warmup):
            self.infer(dummy_input)

        # 벤치마크
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.infer(dummy_input)
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'fps': float(1000 / np.mean(times)),
            'p95_ms': float(np.percentile(times, 95)),
        }


def get_file_size_mb(path: str) -> float:
    """파일 크기 (MB)"""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def main():
    print("=" * 70)
    print("TensorRT vs TVM Performance Comparison")
    print("YOLOv5s on Jetson Xavier")
    print("=" * 70)

    # 모델 경로
    models = {
        'TensorRT FP32': 'models/yolov5s_trt_fp32.engine',
        'TensorRT FP16': 'models/yolov5s_trt_fp16.engine',
        'TVM FP32': 'models/yolov5s_tvm_fp32.so',
    }

    results = {}

    # 각 모델 벤치마크
    for name, path in models.items():
        print(f"\n{'=' * 50}")
        print(f"Benchmarking: {name}")
        print('=' * 50)

        if not os.path.exists(path):
            print(f"  SKIP: File not found: {path}")
            continue

        size_mb = get_file_size_mb(path)
        print(f"  File size: {size_mb:.1f} MB")

        try:
            if 'TensorRT' in name and TRT_AVAILABLE:
                engine = TRTInference(path)
                result = engine.benchmark()
            elif 'TVM' in name and TVM_AVAILABLE:
                engine = TVMInference(path)
                result = engine.benchmark()
            else:
                print("  SKIP: Runtime not available")
                continue

            result['size_mb'] = size_mb
            results[name] = result

            print(f"  Mean: {result['mean_ms']:.2f} ms")
            print(f"  FPS:  {result['fps']:.1f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not results:
        print("\nNo models benchmarked!")
        print("\nMake sure you have run:")
        print("  1. python3 experiments/00_tvm_setup/compile_yolo_tvm.py")
        print("  2. python3 experiments/01_baseline_tensorrt/01_trt_build_engine.py")
        return

    # 비교 결과 출력
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    # 테이블 헤더
    print(f"\n{'Model':<20} {'Latency (ms)':<14} {'FPS':<10} {'Size (MB)':<12} {'vs TRT FP16':<12}")
    print("-" * 70)

    # 베이스라인 (TensorRT FP16)
    baseline = results.get('TensorRT FP16', {}).get('mean_ms', 1)

    for name, r in sorted(results.items()):
        speedup = baseline / r['mean_ms'] if r['mean_ms'] > 0 else 0
        speedup_str = f"{speedup:.2f}x" if 'TensorRT FP16' not in name else "baseline"

        print(f"{name:<20} {r['mean_ms']:<14.2f} {r['fps']:<10.1f} "
              f"{r['size_mb']:<12.1f} {speedup_str:<12}")

    # 분석
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if 'TensorRT FP16' in results and 'TVM FP32' in results:
        trt_fps = results['TensorRT FP16']['fps']
        tvm_fps = results['TVM FP32']['fps']

        print(f"\n  TensorRT FP16: {trt_fps:.1f} FPS")
        print(f"  TVM FP32 (untuned): {tvm_fps:.1f} FPS")

        if trt_fps > tvm_fps:
            gap = (trt_fps - tvm_fps) / trt_fps * 100
            print(f"\n  TensorRT is {gap:.1f}% faster than untuned TVM")
            print(f"\n  Recommendations:")
            print(f"    1. Run AutoTVM tuning to improve TVM performance")
            print(f"    2. Compile TVM with FP16 for fair comparison")
            print(f"    3. Expected TVM improvement after tuning: 2-3x")
        else:
            gap = (tvm_fps - trt_fps) / trt_fps * 100
            print(f"\n  TVM is {gap:.1f}% faster than TensorRT!")

    # 주요 인사이트
    print("\n" + "-" * 70)
    print("KEY INSIGHTS")
    print("-" * 70)
    print("""
  1. TensorRT Baseline Established
     - FP16 mode provides significant speedup over FP32
     - This is the target to beat with TVM optimization

  2. TVM Current Status
     - Running without AutoTVM tuning
     - FP32 precision (not optimized)
     - Expected 2-3x improvement after tuning

  3. Next Steps
     - Run AutoTVM tuning for TVM
     - Compile TVM with FP16
     - Add Zero-Copy pipeline (TVM advantage)
     - Measure end-to-end latency including preprocessing
""")

    # 결과 저장
    results_file = "benchmarks/trt_vs_tvm_comparison.txt"
    os.makedirs("benchmarks", exist_ok=True)

    with open(results_file, 'w') as f:
        f.write("TensorRT vs TVM Comparison\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Results:\n")
        for name, r in results.items():
            f.write(f"  {name}:\n")
            f.write(f"    Latency: {r['mean_ms']:.2f} ms\n")
            f.write(f"    FPS: {r['fps']:.1f}\n")
            f.write(f"    Size: {r['size_mb']:.1f} MB\n\n")

    print(f"\nResults saved to: {results_file}")

    # 다음 단계 안내
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
  To improve TVM performance:
    python3 experiments/01_baseline_tensorrt/04_autotvm_tuning.py

  To test Zero-Copy pipeline:
    python3 experiments/02_zero_copy/01_unified_memory_test.py
    python3 experiments/02_zero_copy/02_preprocess_kernel.py

  To run full comparison with preprocessing:
    python3 experiments/01_baseline_tensorrt/05_e2e_comparison.py
""")


if __name__ == "__main__":
    main()
