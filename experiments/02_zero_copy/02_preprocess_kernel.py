#!/usr/bin/env python3
"""
TVM으로 YOLO 전처리 커널 구현
카메라 프레임 → YOLO 입력 변환을 GPU에서 직접 수행

실행: python3 experiments/02_zero_copy/02_preprocess_kernel.py

변환 과정:
1. Resize: (1080, 1920) → (640, 640)
2. BGR → RGB
3. HWC → CHW
4. Normalize: /255, (x-mean)/std
"""

import time
import numpy as np

import tvm
from tvm import te, tir


class YOLOPreprocessKernel:
    """YOLO 전처리 GPU 커널"""

    def __init__(self, src_h=1080, src_w=1920, dst_h=640, dst_w=640):
        self.src_h = src_h
        self.src_w = src_w
        self.dst_h = dst_h
        self.dst_w = dst_w

        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.kernel = self._build_kernel()

    def _build_kernel(self):
        """전처리 커널 빌드"""

        src_h, src_w = self.src_h, self.src_w
        dst_h, dst_w = self.dst_h, self.dst_w

        # 입력: (H, W, 3) uint8 BGR
        input_img = te.placeholder((src_h, src_w, 3), name="input", dtype="uint8")

        # 스케일 계산
        scale_h = src_h / dst_h
        scale_w = src_w / dst_w

        # 출력: (1, 3, H, W) float32 RGB normalized
        def preprocess(n, c, h, w):
            # 1. Resize (Nearest neighbor for simplicity)
            src_h_idx = tir.Cast("int32", tir.floor(h * scale_h))
            src_w_idx = tir.Cast("int32", tir.floor(w * scale_w))

            # Clamp to valid range
            src_h_idx = tir.max(tir.min(src_h_idx, src_h - 1), 0)
            src_w_idx = tir.max(tir.min(src_w_idx, src_w - 1), 0)

            # 2. BGR → RGB (swap channel 0 and 2)
            src_c = tir.if_then_else(c == 0, 2, tir.if_then_else(c == 2, 0, c))

            # Get pixel value
            pixel = input_img[src_h_idx, src_w_idx, src_c].astype("float32")

            # 3. Normalize: /255, then (x - mean) / std
            pixel = pixel / 255.0

            # Mean and std per channel
            mean_val = tir.Select(c == 0, 0.485,
                        tir.Select(c == 1, 0.456, 0.406))
            std_val = tir.Select(c == 0, 0.229,
                       tir.Select(c == 1, 0.224, 0.225))

            return (pixel - mean_val) / std_val

        output = te.compute(
            (1, 3, dst_h, dst_w),
            preprocess,
            name="preprocessed"
        )

        # 스케줄 생성
        s = te.create_schedule(output.op)

        # GPU 최적화
        n, c, h, w = s[output].op.axis
        fused = s[output].fuse(n, c, h, w)

        # 스레드 블록 설정
        block_size = 256
        bx, tx = s[output].split(fused, factor=block_size)
        s[output].bind(bx, te.thread_axis("blockIdx.x"))
        s[output].bind(tx, te.thread_axis("threadIdx.x"))

        # 빌드
        func = tvm.build(s, [input_img, output], target="cuda -arch=sm_72",
                        name="yolo_preprocess")

        return func

    def __call__(self, input_arr, output_arr):
        """커널 실행"""
        self.kernel(input_arr, output_arr)

    def get_cuda_source(self):
        """생성된 CUDA 코드 반환"""
        return self.kernel.imported_modules[0].get_source()


class YOLOPreprocessKernelBilinear:
    """Bilinear interpolation을 사용하는 고품질 전처리 커널"""

    def __init__(self, src_h=1080, src_w=1920, dst_h=640, dst_w=640):
        self.src_h = src_h
        self.src_w = src_w
        self.dst_h = dst_h
        self.dst_w = dst_w
        self.kernel = self._build_kernel()

    def _build_kernel(self):
        src_h, src_w = self.src_h, self.src_w
        dst_h, dst_w = self.dst_h, self.dst_w

        input_img = te.placeholder((src_h, src_w, 3), name="input", dtype="uint8")

        scale_h = (src_h - 1) / (dst_h - 1)
        scale_w = (src_w - 1) / (dst_w - 1)

        def bilinear_preprocess(n, c, h, w):
            # 원본 좌표 (실수)
            src_h_f = h * scale_h
            src_w_f = w * scale_w

            # 정수 좌표
            h0 = tir.Cast("int32", tir.floor(src_h_f))
            w0 = tir.Cast("int32", tir.floor(src_w_f))
            h1 = tir.min(h0 + 1, src_h - 1)
            w1 = tir.min(w0 + 1, src_w - 1)

            # 보간 가중치
            h_weight = src_h_f - tir.Cast("float32", h0)
            w_weight = src_w_f - tir.Cast("float32", w0)

            # BGR → RGB
            src_c = tir.if_then_else(c == 0, 2, tir.if_then_else(c == 2, 0, c))

            # 4개 픽셀 가져오기
            p00 = input_img[h0, w0, src_c].astype("float32")
            p01 = input_img[h0, w1, src_c].astype("float32")
            p10 = input_img[h1, w0, src_c].astype("float32")
            p11 = input_img[h1, w1, src_c].astype("float32")

            # Bilinear interpolation
            pixel = (p00 * (1 - h_weight) * (1 - w_weight) +
                    p01 * (1 - h_weight) * w_weight +
                    p10 * h_weight * (1 - w_weight) +
                    p11 * h_weight * w_weight)

            # Normalize
            pixel = pixel / 255.0
            mean_val = tir.Select(c == 0, 0.485,
                        tir.Select(c == 1, 0.456, 0.406))
            std_val = tir.Select(c == 0, 0.229,
                       tir.Select(c == 1, 0.224, 0.225))

            return (pixel - mean_val) / std_val

        output = te.compute(
            (1, 3, dst_h, dst_w),
            bilinear_preprocess,
            name="preprocessed_bilinear"
        )

        s = te.create_schedule(output.op)
        n, c, h, w = s[output].op.axis
        fused = s[output].fuse(n, c, h, w)
        bx, tx = s[output].split(fused, factor=256)
        s[output].bind(bx, te.thread_axis("blockIdx.x"))
        s[output].bind(tx, te.thread_axis("threadIdx.x"))

        return tvm.build(s, [input_img, output], target="cuda -arch=sm_72",
                        name="yolo_preprocess_bilinear")

    def __call__(self, input_arr, output_arr):
        self.kernel(input_arr, output_arr)


def benchmark_preprocess_kernel():
    """전처리 커널 벤치마크"""
    print("\n[Benchmark] Preprocess Kernel Comparison")
    print("=" * 50)

    dev = tvm.cuda()

    # 입력 데이터 (Full HD BGR 이미지)
    src_h, src_w = 1080, 1920
    dst_h, dst_w = 640, 640

    input_data = np.random.randint(0, 256, (src_h, src_w, 3), dtype=np.uint8)
    input_gpu = tvm.nd.array(input_data, dev)
    output_gpu = tvm.nd.empty((1, 3, dst_h, dst_w), "float32", dev)

    # 1. Nearest Neighbor 커널
    print("\n[1] Nearest Neighbor Kernel")
    print("-" * 40)
    kernel_nn = YOLOPreprocessKernel(src_h, src_w, dst_h, dst_w)

    # 워밍업
    for _ in range(10):
        kernel_nn(input_gpu, output_gpu)
    dev.sync()

    times = []
    for _ in range(100):
        start = time.time()
        kernel_nn(input_gpu, output_gpu)
        dev.sync()
        times.append((time.time() - start) * 1000)

    print(f"  Input:  ({src_h}, {src_w}, 3) uint8")
    print(f"  Output: (1, 3, {dst_h}, {dst_w}) float32")
    print(f"  Time: {np.mean(times):.3f} ms (±{np.std(times):.3f})")

    # 2. Bilinear 커널
    print("\n[2] Bilinear Interpolation Kernel")
    print("-" * 40)
    kernel_bilinear = YOLOPreprocessKernelBilinear(src_h, src_w, dst_h, dst_w)

    for _ in range(10):
        kernel_bilinear(input_gpu, output_gpu)
    dev.sync()

    times_bilinear = []
    for _ in range(100):
        start = time.time()
        kernel_bilinear(input_gpu, output_gpu)
        dev.sync()
        times_bilinear.append((time.time() - start) * 1000)

    print(f"  Time: {np.mean(times_bilinear):.3f} ms (±{np.std(times_bilinear):.3f})")

    # 3. CPU 비교 (OpenCV 스타일)
    print("\n[3] CPU Baseline (NumPy)")
    print("-" * 40)

    import cv2
    times_cpu = []
    for _ in range(20):
        start = time.time()
        # Resize
        resized = cv2.resize(input_data, (dst_w, dst_h))
        # BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # HWC → CHW
        chw = rgb.transpose(2, 0, 1)
        # Normalize
        normalized = (chw.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
        # Add batch dim
        output_cpu = normalized[np.newaxis, ...]
        times_cpu.append((time.time() - start) * 1000)

    print(f"  Time: {np.mean(times_cpu):.3f} ms (±{np.std(times_cpu):.3f})")

    # 결과 비교
    print("\n" + "=" * 50)
    print("Comparison Summary")
    print("=" * 50)
    print(f"  GPU (Nearest):  {np.mean(times):.3f} ms")
    print(f"  GPU (Bilinear): {np.mean(times_bilinear):.3f} ms")
    print(f"  CPU (OpenCV):   {np.mean(times_cpu):.3f} ms")
    print(f"  Speedup (NN):   {np.mean(times_cpu) / np.mean(times):.1f}x")
    print(f"  Speedup (Bil):  {np.mean(times_cpu) / np.mean(times_bilinear):.1f}x")


def show_generated_code():
    """생성된 CUDA 코드 표시"""
    print("\n[Generated CUDA Code]")
    print("=" * 50)

    kernel = YOLOPreprocessKernel()
    cuda_code = kernel.get_cuda_source()

    # 첫 80줄만 출력
    lines = cuda_code.split('\n')[:80]
    print('\n'.join(lines))
    print("...")


def main():
    print("=" * 50)
    print("YOLO Preprocessing Kernel (TVM)")
    print("=" * 50)

    print("\n** Zero-Copy 전처리 핵심 **")
    print("  - 카메라 버퍼 → GPU 전처리 → YOLO 입력")
    print("  - CPU 전처리 제거 → 복사 병목 해결")
    print("  - TVM으로 커스텀 커널 작성 → 완전 제어")

    # 벤치마크
    benchmark_preprocess_kernel()

    # 생성된 코드 확인
    show_generated_code()

    print("\n" + "=" * 50)
    print("Next Steps")
    print("=" * 50)
    print("  1. 후처리(NMS) 커널 구현")
    print("  2. 전체 파이프라인 통합")
    print("  3. End-to-End 벤치마크")


if __name__ == "__main__":
    main()
