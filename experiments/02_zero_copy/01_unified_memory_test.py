#!/usr/bin/env python3
"""
Jetson Xavier Unified Memory 테스트
Zero-Copy 파이프라인의 기초

실행: python3 experiments/02_zero_copy/01_unified_memory_test.py

핵심 개념:
- Jetson Xavier는 CPU/GPU가 물리 메모리 공유 (Unified Memory)
- cudaMallocManaged로 할당된 메모리는 복사 없이 양쪽에서 접근 가능
- TVM의 tvm.nd.array는 기본적으로 디바이스별 메모리 할당
- Jetson에서는 이를 활용한 Zero-Copy 가능
"""

import time
import numpy as np
import ctypes

import tvm
from tvm import te


def test_basic_tvm_array():
    """기본 TVM NDArray 테스트"""
    print("\n[1/4] Basic TVM NDArray")
    print("-" * 40)

    # CPU 배열
    cpu_arr = tvm.nd.array(np.random.randn(1000, 1000).astype("float32"))
    print(f"  CPU Array: {cpu_arr.shape}, device={cpu_arr.device}")

    # GPU 배열
    gpu_arr = tvm.nd.array(np.random.randn(1000, 1000).astype("float32"), tvm.cuda())
    print(f"  GPU Array: {gpu_arr.shape}, device={gpu_arr.device}")

    # CPU → GPU 복사
    start = time.time()
    gpu_arr2 = tvm.nd.array(cpu_arr.numpy(), tvm.cuda())
    copy_time = (time.time() - start) * 1000
    print(f"  CPU→GPU Copy Time: {copy_time:.2f} ms")

    return copy_time


def test_unified_memory_concept():
    """Unified Memory 개념 테스트 (Jetson 특화)"""
    print("\n[2/4] Unified Memory Concept on Jetson")
    print("-" * 40)

    # Jetson에서는 CPU/GPU가 같은 물리 메모리 사용
    # tvm.cuda() 배열도 사실상 Unified Memory에 있음

    # 큰 배열로 테스트
    shape = (1, 3, 640, 640)  # YOLO 입력 크기
    size_mb = np.prod(shape) * 4 / 1024 / 1024

    # 방법 1: NumPy → TVM GPU (복사 발생할 수 있음)
    np_arr = np.random.randn(*shape).astype("float32")

    start = time.time()
    gpu_arr1 = tvm.nd.array(np_arr, tvm.cuda())
    time1 = (time.time() - start) * 1000

    # 방법 2: TVM GPU 직접 생성 후 데이터 채우기
    start = time.time()
    gpu_arr2 = tvm.nd.empty(shape, "float32", tvm.cuda())
    # 데이터 복사
    np.copyto(gpu_arr2.numpy(), np_arr)  # Jetson에서는 이게 빠를 수 있음
    time2 = (time.time() - start) * 1000

    print(f"  Array Size: {size_mb:.1f} MB")
    print(f"  Method 1 (np→gpu): {time1:.2f} ms")
    print(f"  Method 2 (empty+copy): {time2:.2f} ms")

    # GPU 접근 테스트
    start = time.time()
    _ = gpu_arr1.numpy()  # GPU → NumPy
    access_time = (time.time() - start) * 1000
    print(f"  GPU→NumPy Access: {access_time:.2f} ms")


def test_zero_copy_kernel():
    """Zero-Copy 커널 테스트"""
    print("\n[3/4] Zero-Copy Kernel Execution")
    print("-" * 40)

    # 간단한 전처리 커널: 이미지 정규화
    # input (H, W, 3) uint8 → output (3, H, W) float32 normalized

    H, W = 640, 640

    # 커널 정의
    input_img = te.placeholder((H, W, 3), name="input", dtype="uint8")

    # HWC → CHW + Normalize
    def normalize(c, h, w):
        pixel = input_img[h, w, c].astype("float32")
        return pixel / 255.0

    output = te.compute((3, H, W), normalize, name="output")

    # 스케줄
    s = te.create_schedule(output.op)
    c, h, w = s[output].op.axis
    fused = s[output].fuse(c, h, w)
    bx, tx = s[output].split(fused, factor=256)
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))

    # 빌드
    func = tvm.build(s, [input_img, output], target="cuda -arch=sm_72", name="normalize")

    # 데이터 준비 (Unified Memory에서)
    dev = tvm.cuda()

    # 방법 A: 전통적인 방식 (복사 포함)
    np_input = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    gpu_input = tvm.nd.array(np_input, dev)
    gpu_output = tvm.nd.empty((3, H, W), "float32", dev)

    # 워밍업
    for _ in range(10):
        func(gpu_input, gpu_output)
    dev.sync()

    # 벤치마크
    times = []
    for _ in range(100):
        start = time.time()
        func(gpu_input, gpu_output)
        dev.sync()
        times.append((time.time() - start) * 1000)

    print(f"  Normalize Kernel (GPU):")
    print(f"    Input:  ({H}, {W}, 3) uint8")
    print(f"    Output: (3, {H}, {W}) float32")
    print(f"    Time: {np.mean(times):.3f} ms (±{np.std(times):.3f})")

    # 결과 검증
    expected = np_input.transpose(2, 0, 1).astype(np.float32) / 255.0
    np.testing.assert_allclose(gpu_output.numpy(), expected, rtol=1e-5)
    print(f"    Correctness: VERIFIED")


def test_memory_bandwidth():
    """메모리 대역폭 측정"""
    print("\n[4/4] Memory Bandwidth Test")
    print("-" * 40)

    # 다양한 크기로 복사 속도 측정
    sizes = [
        (1, 3, 224, 224),    # ResNet 입력
        (1, 3, 640, 640),    # YOLO 입력
        (1, 3, 1080, 1920),  # Full HD
    ]

    dev = tvm.cuda()

    for shape in sizes:
        size_mb = np.prod(shape) * 4 / 1024 / 1024

        # CPU → GPU
        np_data = np.random.randn(*shape).astype("float32")

        times = []
        for _ in range(20):
            start = time.time()
            gpu_arr = tvm.nd.array(np_data, dev)
            dev.sync()
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        bandwidth = size_mb / (avg_time / 1000)

        print(f"  Shape {shape}:")
        print(f"    Size: {size_mb:.1f} MB")
        print(f"    Time: {avg_time:.2f} ms")
        print(f"    Bandwidth: {bandwidth:.1f} MB/s")


def main():
    print("=" * 50)
    print("Jetson Xavier Unified Memory Test")
    print("=" * 50)

    print("\n** Jetson Xavier Unified Memory 특성 **")
    print("  - CPU와 GPU가 같은 물리 메모리 공유")
    print("  - cudaMallocManaged → 자동 페이지 마이그레이션")
    print("  - Zero-Copy: 데이터 이동 없이 양쪽에서 접근")
    print("  - 이 특성을 활용해 파이프라인 최적화 가능")

    # 테스트 실행
    test_basic_tvm_array()
    test_unified_memory_concept()
    test_zero_copy_kernel()
    test_memory_bandwidth()

    print("\n" + "=" * 50)
    print("Key Takeaways")
    print("=" * 50)
    print("  1. Jetson에서 TVM GPU 배열은 Unified Memory 활용")
    print("  2. CPU↔GPU 복사가 불필요하거나 빠름")
    print("  3. 전처리/추론/후처리를 모두 GPU에서 실행 시 최적")
    print("  4. 다음 단계: 전처리 커널 확장 구현")
    print("=" * 50)

    print("\nNext: Full preprocessing kernel")
    print("  python3 experiments/02_zero_copy/02_preprocess_kernel.py")


if __name__ == "__main__":
    main()
