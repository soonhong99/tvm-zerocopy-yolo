#!/usr/bin/env python3
"""
TensorRT 엔진 빌드 스크립트
ONNX 모델을 TensorRT 엔진으로 변환

실행: python3 experiments/01_baseline_tensorrt/01_trt_build_engine.py

출력:
- models/yolov5s_trt_fp32.engine
- models/yolov5s_trt_fp16.engine
"""

import os
import sys
import time
import numpy as np

# TensorRT import
try:
    import tensorrt as trt
except ImportError:
    print("ERROR: TensorRT not found!")
    print("  JetPack should include TensorRT.")
    print("  Try: python3 -c 'import tensorrt; print(tensorrt.__version__)'")
    sys.exit(1)


# TensorRT Logger
class TRTLogger(trt.Logger):
    def __init__(self, verbose=False):
        super().__init__(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str, precision: str = "fp32",
                 workspace_gb: float = 2.0, verbose: bool = False) -> bool:
    """
    ONNX 모델을 TensorRT 엔진으로 변환

    Args:
        onnx_path: ONNX 모델 경로
        engine_path: 출력 엔진 경로
        precision: "fp32", "fp16", or "int8"
        workspace_gb: GPU 워크스페이스 크기 (GB)
        verbose: 상세 로그 출력

    Returns:
        성공 여부
    """
    logger = TRTLogger(verbose)

    print(f"\n[Building TensorRT Engine]")
    print(f"  ONNX: {onnx_path}")
    print(f"  Engine: {engine_path}")
    print(f"  Precision: {precision.upper()}")
    print("-" * 50)

    # Builder 생성
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    # ONNX 파싱
    print("  Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX!")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return False

    print(f"  Network inputs: {network.num_inputs}")
    print(f"  Network outputs: {network.num_outputs}")

    # Input shape 확인
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    Input[{i}]: {inp.name}, shape={inp.shape}, dtype={inp.dtype}")

    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    Output[{i}]: {out.name}, shape={out.shape}, dtype={out.dtype}")

    # Config 설정
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    # 정밀도 설정
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 mode enabled")
        else:
            print("  WARNING: FP16 not supported, falling back to FP32")

    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # INT8은 calibration 필요 - 여기서는 FP16 fallback
            config.set_flag(trt.BuilderFlag.FP16)
            print("  INT8 mode enabled (with FP16 fallback)")
        else:
            print("  WARNING: INT8 not supported, falling back to FP16/FP32")

    # 엔진 빌드
    print("\n  Building engine (this may take several minutes)...")
    start_time = time.time()

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build engine!")
        return False

    build_time = time.time() - start_time
    print(f"  Build time: {build_time:.1f} seconds")

    # 엔진 저장
    print(f"  Saving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  Engine size: {engine_size_mb:.1f} MB")

    print("\n  SUCCESS!")
    return True


def main():
    print("=" * 60)
    print("TensorRT Engine Builder for YOLOv5s")
    print("=" * 60)

    # TensorRT 버전 확인
    print(f"\nTensorRT version: {trt.__version__}")

    # 경로 설정
    onnx_path = "models/yolov5s.onnx"
    if not os.path.exists(onnx_path):
        print(f"\nERROR: ONNX model not found: {onnx_path}")
        print("  Run this first:")
        print("  python3 experiments/00_tvm_setup/export_yolo_onnx.py")
        sys.exit(1)

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nONNX model: {onnx_path} ({onnx_size_mb:.1f} MB)")

    # FP32 엔진 빌드
    print("\n" + "=" * 60)
    print("Building FP32 Engine")
    print("=" * 60)
    fp32_path = "models/yolov5s_trt_fp32.engine"
    build_engine(onnx_path, fp32_path, precision="fp32")

    # FP16 엔진 빌드
    print("\n" + "=" * 60)
    print("Building FP16 Engine")
    print("=" * 60)
    fp16_path = "models/yolov5s_trt_fp16.engine"
    build_engine(onnx_path, fp16_path, precision="fp16")

    # 결과 요약
    print("\n" + "=" * 60)
    print("Build Summary")
    print("=" * 60)

    for path in [fp32_path, fp16_path]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {path}: {size_mb:.1f} MB")
        else:
            print(f"  {path}: FAILED")

    print("\nNext step:")
    print("  python3 experiments/01_baseline_tensorrt/02_trt_benchmark.py")


if __name__ == "__main__":
    main()
