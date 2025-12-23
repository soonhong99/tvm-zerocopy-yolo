#!/usr/bin/env python3
"""
Compile YOLOv5 ONNX model with TVM
"""
import os
import time
import numpy as np
import onnx
import tvm
from tvm import relay
from tvm.contrib import graph_executor


def compile_yolo_tvm(onnx_path="models/yolov5s.onnx"):
    """Compile YOLO ONNX model with TVM"""

    print("="*50)
    print("YOLOv5 TVM Compilation")
    print("="*50)

    # Load ONNX
    print("\n[1/3] Loading ONNX model...")
    onnx_model = onnx.load(onnx_path)

    # Convert to Relay IR
    print("[2/3] Converting to Relay IR and compiling...")
    input_shape = {"images": [1, 3, 640, 640]}
    mod, params = relay.frontend.from_onnx(onnx_model, shape=input_shape)

    print(f"  Input shape: {input_shape}")
    print(f"  Num params: {len(params)}")

    # Compile for Xavier
    target = tvm.target.Target("cuda -arch=sm_72")

    start_time = time.time()
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    compile_time = time.time() - start_time

    print(f"  Compile time: {compile_time:.1f} sec")

    # Benchmark
    print("\n[3/3] Benchmarking...")
    dev = tvm.cuda()
    module = graph_executor.GraphModule(lib["default"](dev))

    # Prepare input
    input_data = np.random.randn(1, 3, 640, 640).astype("float32")
    module.set_input("images", input_data)

    # Warm up
    print("  Warming up...")
    for _ in range(20):
        module.run()
    dev.sync()

    # Measure
    print("  Measuring...")
    num_runs = 100
    times = []
    for _ in range(num_runs):
        start = time.time()
        module.run()
        dev.sync()
        times.append((time.time() - start) * 1000)

    # Get outputs
    num_outputs = module.get_num_outputs()
    output_shapes = [module.get_output(i).shape for i in range(num_outputs)]

    # Results
    mean_time = np.mean(times)
    fps = 1000 / mean_time

    print(f"\n  Num outputs: {num_outputs}")
    for i, shape in enumerate(output_shapes):
        print(f"  Output[{i}] shape: {shape}")

    print(f"\n  Inference time:")
    print(f"    Mean: {mean_time:.2f} ms")
    print(f"    Std:  {np.std(times):.2f} ms")
    print(f"    Min:  {np.min(times):.2f} ms")
    print(f"    Max:  {np.max(times):.2f} ms")
    print(f"  FPS: {fps:.1f}")

    # Save compiled model
    output_path = "models/yolov5s_tvm_fp32.so"
    os.makedirs("models", exist_ok=True)
    lib.export_library(output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024

    print(f"\n  Saved compiled model:")
    print(f"    Path: {output_path}")
    print(f"    Size: {size_mb:.1f} MB")

    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"  Model: YOLOv5s")
    print(f"  Input: [1, 3, 640, 640]")
    print(f"  Precision: FP32")
    print(f"  Inference: {mean_time:.2f} ms ({fps:.1f} FPS)")
    print(f"  Compiled: {output_path}")
    print("="*50)

    return {
        "mean_ms": mean_time,
        "fps": fps,
        "output_shapes": output_shapes,
        "compiled_path": output_path
    }


if __name__ == "__main__":
    compile_yolo_tvm()
