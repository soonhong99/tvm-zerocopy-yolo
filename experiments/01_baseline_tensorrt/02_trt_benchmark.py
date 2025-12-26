#!/usr/bin/env python3
"""
TensorRT 벤치마크 스크립트
TensorRT 엔진 성능 측정

실행: python3 experiments/01_baseline_tensorrt/02_trt_benchmark.py

측정 항목:
- 추론 시간 (ms)
- FPS
- GPU 메모리 사용량
"""

import os
import sys
import time
import numpy as np
from typing import Dict, Tuple

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"ERROR: {e}")
    print("  Required: tensorrt, pycuda")
    print("  Install: pip3 install pycuda")
    sys.exit(1)


class TRTInference:
    """TensorRT 추론 엔진 래퍼"""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)

        # 엔진 로드
        print(f"Loading engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load engine")

        self.context = self.engine.create_execution_context()

        # 입출력 바인딩 설정
        self._setup_bindings()

    def _setup_bindings(self):
        """입출력 버퍼 설정"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)

            # GPU 메모리 할당
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name,
                    'shape': shape,
                    'dtype': dtype,
                    'device': device_mem,
                    'host': np.empty(shape, dtype=dtype)
                })
            else:
                self.outputs.append({
                    'name': name,
                    'shape': shape,
                    'dtype': dtype,
                    'device': device_mem,
                    'host': np.empty(shape, dtype=dtype)
                })

        print(f"  Inputs: {[(inp['name'], inp['shape']) for inp in self.inputs]}")
        print(f"  Outputs: {[(out['name'], out['shape']) for out in self.outputs]}")

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """추론 실행"""
        # 입력 데이터 복사
        np.copyto(self.inputs[0]['host'], input_data.ravel().reshape(self.inputs[0]['shape']))
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # 텐서 주소 설정 및 추론
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))

        self.context.execute_async_v3(self.stream.handle)

        # 출력 복사
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()
        return self.outputs[0]['host']

    def benchmark(self, num_warmup: int = 20, num_runs: int = 100) -> Dict:
        """벤치마크 실행"""
        # 더미 입력
        input_shape = self.inputs[0]['shape']
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # 워밍업
        print(f"  Warming up ({num_warmup} runs)...")
        for _ in range(num_warmup):
            self.infer(dummy_input)

        # 벤치마크
        print(f"  Benchmarking ({num_runs} runs)...")
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
            'median_ms': float(np.median(times)),
            'fps': float(1000 / np.mean(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
        }


def benchmark_engine(engine_path: str, name: str) -> Dict:
    """단일 엔진 벤치마크"""
    print(f"\n{'=' * 50}")
    print(f"Benchmarking: {name}")
    print('=' * 50)

    if not os.path.exists(engine_path):
        print(f"  ERROR: Engine not found: {engine_path}")
        return None

    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  Engine size: {engine_size_mb:.1f} MB")

    try:
        trt_engine = TRTInference(engine_path)
        results = trt_engine.benchmark(num_warmup=20, num_runs=100)
        results['engine_size_mb'] = engine_size_mb
        results['engine_path'] = engine_path
        return results
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    print("=" * 60)
    print("TensorRT YOLOv5s Benchmark")
    print("=" * 60)

    print(f"\nTensorRT version: {trt.__version__}")

    # 엔진 목록
    engines = [
        ("models/yolov5s_trt_fp32.engine", "TensorRT FP32"),
        ("models/yolov5s_trt_fp16.engine", "TensorRT FP16"),
    ]

    results = {}
    for engine_path, name in engines:
        result = benchmark_engine(engine_path, name)
        if result:
            results[name] = result

    # 결과 출력
    print("\n" + "=" * 60)
    print("Benchmark Results Summary")
    print("=" * 60)

    if not results:
        print("\nNo engines benchmarked!")
        print("Run first: python3 experiments/01_baseline_tensorrt/01_trt_build_engine.py")
        return

    print(f"\n{'Engine':<20} {'Mean (ms)':<12} {'Std (ms)':<10} {'FPS':<10} {'P95 (ms)':<10}")
    print("-" * 62)

    for name, r in results.items():
        print(f"{name:<20} {r['mean_ms']:<12.2f} {r['std_ms']:<10.2f} "
              f"{r['fps']:<10.1f} {r['p95_ms']:<10.2f}")

    # 상세 결과
    print("\n" + "-" * 60)
    print("Detailed Results")
    print("-" * 60)

    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Latency:")
        print(f"    Mean:   {r['mean_ms']:.2f} ms")
        print(f"    Std:    {r['std_ms']:.2f} ms")
        print(f"    Min:    {r['min_ms']:.2f} ms")
        print(f"    Max:    {r['max_ms']:.2f} ms")
        print(f"    Median: {r['median_ms']:.2f} ms")
        print(f"    P95:    {r['p95_ms']:.2f} ms")
        print(f"    P99:    {r['p99_ms']:.2f} ms")
        print(f"  Throughput:")
        print(f"    FPS:    {r['fps']:.1f}")
        print(f"  Engine Size: {r['engine_size_mb']:.1f} MB")

    # 결과 저장
    results_file = "benchmarks/tensorrt_baseline.txt"
    os.makedirs("benchmarks", exist_ok=True)

    with open(results_file, 'w') as f:
        f.write("TensorRT YOLOv5s Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"TensorRT Version: {trt.__version__}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for name, r in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Mean: {r['mean_ms']:.2f} ms\n")
            f.write(f"  FPS:  {r['fps']:.1f}\n")
            f.write(f"  P95:  {r['p95_ms']:.2f} ms\n\n")

    print(f"\nResults saved to: {results_file}")

    print("\nNext step:")
    print("  python3 experiments/01_baseline_tensorrt/03_compare_trt_tvm.py")


if __name__ == "__main__":
    main()
