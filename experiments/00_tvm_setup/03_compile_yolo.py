#!/usr/bin/env python3
"""
YOLOv5 TVM 컴파일
실행: python3 experiments/00_tvm_setup/03_compile_yolo.py

Prerequisites:
  pip3 install ultralytics onnx onnxruntime
"""

import os
import time
import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_executor


def download_yolo_onnx(model_name="yolov5s", output_dir="models"):
    """YOLOv5 ONNX 모델 다운로드/변환"""
    print("\n[1/5] Preparing YOLO ONNX Model")
    print("-" * 40)

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    if os.path.exists(onnx_path):
        print(f"  Found existing: {onnx_path}")
        return onnx_path

    try:
        # Ultralytics에서 ONNX로 export
        from ultralytics import YOLO

        model = YOLO(f"{model_name}.pt")
        model.export(format="onnx", imgsz=640, simplify=True)

        # 이동
        src = f"{model_name}.onnx"
        if os.path.exists(src):
            os.rename(src, onnx_path)
            print(f"  Exported: {onnx_path}")
    except ImportError:
        print("  ultralytics not installed, trying torch hub...")

        import torch
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        model.eval()

        # ONNX export
        dummy_input = torch.randn(1, 3, 640, 640)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=12,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        print(f"  Exported via torch hub: {onnx_path}")

    return onnx_path


def load_onnx_model(onnx_path):
    """ONNX → Relay IR 변환"""
    print("\n[2/5] Converting ONNX to Relay IR")
    print("-" * 40)

    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # 입력 shape (batch=1, C=3, H=640, W=640)
    input_shape = {"images": [1, 3, 640, 640]}

    mod, params = relay.frontend.from_onnx(onnx_model, shape=input_shape)

    print(f"  ONNX Path: {onnx_path}")
    print(f"  Input Shape: {input_shape}")
    print(f"  Num Params: {len(params)}")

    return mod, params


def apply_optimizations(mod, params):
    """그래프 최적화 패스 적용"""
    print("\n[3/5] Applying Optimizations")
    print("-" * 40)

    # 최적화 전 연산 수
    before_ops = count_ops(mod)

    with tvm.transform.PassContext(opt_level=3):
        # 표준 최적화
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.FuseOps(fuse_opt_level=2)(mod)
        mod = relay.transform.SimplifyInference()(mod)

    # 최적화 후 연산 수
    after_ops = count_ops(mod)

    print(f"  Ops before: {before_ops}")
    print(f"  Ops after:  {after_ops}")
    print(f"  Reduction:  {before_ops - after_ops} ops fused/removed")

    return mod


def count_ops(mod):
    """Relay IR의 연산 수 카운트"""
    class OpCounter(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            self.count += 1
            super().visit_call(call)

    counter = OpCounter()
    counter.visit(mod["main"])
    return counter.count


def compile_for_xavier(mod, params, precision="fp32"):
    """Xavier GPU용 컴파일"""
    print(f"\n[4/5] Compiling for Xavier ({precision})")
    print("-" * 40)

    target = tvm.target.Target("cuda -arch=sm_72")

    # FP16 변환 (선택적)
    if precision == "fp16":
        mod = relay.transform.ToMixedPrecision("float16")(mod)
        print("  Applied FP16 mixed precision")

    start = time.time()
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    compile_time = time.time() - start

    print(f"  Precision: {precision}")
    print(f"  Compile Time: {compile_time:.1f} sec")

    return lib


def benchmark_yolo(lib, num_runs=100):
    """YOLO 추론 벤치마크"""
    print(f"\n[5/5] Benchmarking YOLO ({num_runs} runs)")
    print("-" * 40)

    dev = tvm.cuda()
    module = graph_executor.GraphModule(lib["default"](dev))

    # 입력 데이터
    input_data = np.random.randn(1, 3, 640, 640).astype("float32")
    module.set_input("images", input_data)

    # 워밍업
    print("  Warming up...")
    for _ in range(20):
        module.run()
    dev.sync()

    # 벤치마크
    print("  Measuring...")
    times = []
    for _ in range(num_runs):
        start = time.time()
        module.run()
        dev.sync()
        times.append((time.time() - start) * 1000)

    # 출력 확인
    num_outputs = module.get_num_outputs()
    output_shapes = [module.get_output(i).shape for i in range(num_outputs)]

    print(f"  Num Outputs: {num_outputs}")
    for i, shape in enumerate(output_shapes):
        print(f"  Output[{i}] Shape: {shape}")

    print(f"\n  Inference Time:")
    print(f"    Mean: {np.mean(times):.2f} ms")
    print(f"    Std:  {np.std(times):.2f} ms")
    print(f"    Min:  {np.min(times):.2f} ms")
    print(f"    Max:  {np.max(times):.2f} ms")
    print(f"  FPS: {1000 / np.mean(times):.1f}")

    return {
        "mean_ms": np.mean(times),
        "fps": 1000 / np.mean(times),
        "output_shapes": output_shapes
    }


def save_compiled_model(lib, output_path):
    """컴파일된 모델 저장"""
    print(f"\n[Bonus] Saving compiled model")
    print("-" * 40)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 저장
    lib.export_library(output_path)
    print(f"  Saved to: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def main():
    print("=" * 50)
    print("YOLOv5 TVM Compilation")
    print("=" * 50)

    # 1. ONNX 모델 준비
    onnx_path = download_yolo_onnx("yolov5s", output_dir="models")

    # 2. Relay IR 변환
    mod, params = load_onnx_model(onnx_path)

    # 3. 최적화 적용
    mod = apply_optimizations(mod, params)

    # 4. 컴파일 (FP32)
    lib_fp32 = compile_for_xavier(mod, params, precision="fp32")

    # 5. 벤치마크
    results = benchmark_yolo(lib_fp32)

    # 6. 저장
    save_compiled_model(lib_fp32, "models/yolov5s_tvm_fp32.so")

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  Model: YOLOv5s")
    print(f"  Input: [1, 3, 640, 640]")
    print(f"  Precision: FP32")
    print(f"  Inference: {results['mean_ms']:.2f} ms ({results['fps']:.1f} FPS)")
    print("=" * 50)

    print("\n** 이제 Zero-Copy 파이프라인 구현 시작! **")
    print("  python3 experiments/02_zero_copy/01_unified_memory_test.py")


if __name__ == "__main__":
    main()
