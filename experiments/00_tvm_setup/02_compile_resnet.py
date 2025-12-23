#!/usr/bin/env python3
"""
ResNet18로 TVM 컴파일 파이프라인 학습
YOLOv5 전에 간단한 모델로 연습

실행: python3 experiments/00_tvm_setup/02_compile_resnet.py
"""

import time
import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_executor


def load_pytorch_model():
    """PyTorch ResNet18 로드"""
    print("\n[1/4] Loading PyTorch Model")
    print("-" * 40)

    import torch
    import torchvision

    # 모델 로드
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    # 입력 shape
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)

    # TorchScript 변환
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)

    print(f"  Model: ResNet18")
    print(f"  Input Shape: {input_shape}")

    return scripted_model, input_shape


def convert_to_relay(scripted_model, input_shape):
    """TorchScript → Relay IR 변환"""
    print("\n[2/4] Converting to Relay IR")
    print("-" * 40)

    input_name = "input0"
    shape_list = [(input_name, input_shape)]

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # Relay IR 출력 (일부만)
    relay_text = str(mod)
    print(f"  Relay IR (first 500 chars):")
    print(f"  {relay_text[:500]}...")
    print(f"  Total length: {len(relay_text)} chars")
    print(f"  Num params: {len(params)}")

    return mod, params, input_name


def compile_model(mod, params, opt_level=3):
    """Relay → 컴파일된 라이브러리"""
    print(f"\n[3/4] Compiling with opt_level={opt_level}")
    print("-" * 40)

    # Xavier GPU 타겟
    target = tvm.target.Target("cuda -arch=sm_72")

    start = time.time()
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target=target, params=params)
    compile_time = time.time() - start

    print(f"  Target: cuda -arch=sm_72 (Jetson Xavier)")
    print(f"  Opt Level: {opt_level}")
    print(f"  Compile Time: {compile_time:.1f} sec")

    return lib


def benchmark_inference(lib, input_name, input_shape, num_runs=100):
    """추론 성능 측정"""
    print(f"\n[4/4] Benchmarking ({num_runs} runs)")
    print("-" * 40)

    dev = tvm.cuda()
    module = graph_executor.GraphModule(lib["default"](dev))

    # 입력 데이터
    input_data = np.random.randn(*input_shape).astype("float32")
    module.set_input(input_name, input_data)

    # 워밍업
    print("  Warming up...")
    for _ in range(10):
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

    # 결과
    output = module.get_output(0).numpy()

    print(f"  Output Shape: {output.shape}")
    print(f"  Inference Time:")
    print(f"    Mean: {np.mean(times):.2f} ms")
    print(f"    Std:  {np.std(times):.2f} ms")
    print(f"    Min:  {np.min(times):.2f} ms")
    print(f"    Max:  {np.max(times):.2f} ms")
    print(f"  FPS: {1000 / np.mean(times):.1f}")

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "fps": 1000 / np.mean(times)
    }


def show_generated_code(lib):
    """생성된 CUDA 코드 일부 출력"""
    print("\n[Bonus] Generated CUDA Code (sample)")
    print("-" * 40)

    try:
        cuda_code = lib.imported_modules[0].get_source()
        # 첫 번째 커널만 출력
        lines = cuda_code.split('\n')[:50]
        print('\n'.join(lines))
        print("  ...")
    except Exception as e:
        print(f"  Could not retrieve CUDA code: {e}")


def main():
    print("=" * 50)
    print("TVM ResNet18 Compilation Tutorial")
    print("=" * 50)

    # 1. PyTorch 모델 로드
    scripted_model, input_shape = load_pytorch_model()

    # 2. Relay IR 변환
    mod, params, input_name = convert_to_relay(scripted_model, input_shape)

    # 3. 컴파일
    lib = compile_model(mod, params, opt_level=3)

    # 4. 벤치마크
    results = benchmark_inference(lib, input_name, input_shape)

    # 5. 생성된 코드 확인
    show_generated_code(lib)

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  Model: ResNet18")
    print(f"  Input: {input_shape}")
    print(f"  Inference: {results['mean_ms']:.2f} ms ({results['fps']:.1f} FPS)")
    print("=" * 50)

    print("\nNext: Try compiling YOLOv5 with:")
    print("  python3 experiments/00_tvm_setup/03_compile_yolo.py")


if __name__ == "__main__":
    main()
