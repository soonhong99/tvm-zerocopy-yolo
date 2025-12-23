#!/usr/bin/env python3
"""
INT8/FP16/FP32 양자화 비교 실험
Adaptive Precision의 기초

실행: python3 experiments/03_adaptive_precision/01_quantization_compare.py

목표:
- 각 정밀도별 성능/정확도 트레이드오프 분석
- 런타임 정밀도 전환 가능성 검증
"""

import os
import time
import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_executor


class MultiPrecisionModel:
    """다중 정밀도 모델 컨테이너"""

    def __init__(self, model_path=None):
        self.models = {}
        self.compile_times = {}

        if model_path:
            self.load_and_compile(model_path)

    def load_and_compile(self, model_path):
        """모델 로드 및 다중 정밀도 컴파일"""
        print("\n[Loading Model]")
        print("-" * 40)

        # ONNX 로드
        import onnx
        onnx_model = onnx.load(model_path)

        input_shape = {"images": [1, 3, 640, 640]}
        mod, params = relay.frontend.from_onnx(onnx_model, shape=input_shape)

        target = tvm.target.Target("cuda -arch=sm_72")

        # FP32 컴파일
        print("\n  Compiling FP32...")
        self.models["fp32"], self.compile_times["fp32"] = \
            self._compile_fp32(mod, params, target)

        # FP16 컴파일
        print("  Compiling FP16...")
        self.models["fp16"], self.compile_times["fp16"] = \
            self._compile_fp16(mod, params, target)

        # INT8 컴파일 (calibration 필요)
        print("  Compiling INT8 (simulated)...")
        self.models["int8"], self.compile_times["int8"] = \
            self._compile_int8(mod, params, target)

    def _compile_fp32(self, mod, params, target):
        start = time.time()
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        compile_time = time.time() - start

        dev = tvm.cuda()
        module = graph_executor.GraphModule(lib["default"](dev))
        return module, compile_time

    def _compile_fp16(self, mod, params, target):
        start = time.time()

        # FP16 변환
        mod_fp16 = relay.transform.InferType()(mod)
        mod_fp16 = relay.transform.ToMixedPrecision("float16")(mod_fp16)

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod_fp16, target=target, params=params)
        compile_time = time.time() - start

        dev = tvm.cuda()
        module = graph_executor.GraphModule(lib["default"](dev))
        return module, compile_time

    def _compile_int8(self, mod, params, target):
        """INT8 양자화 (시뮬레이션)

        실제 INT8은 calibration dataset 필요.
        여기서는 개념적 테스트만 수행.
        """
        start = time.time()

        # 실제 INT8은 이렇게:
        # with relay.quantize.qconfig(calibrate_mode="global_scale",
        #                             global_scale=8.0):
        #     mod_int8 = relay.quantize.quantize(mod, params)

        # 시뮬레이션: FP32 모델 사용
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        compile_time = time.time() - start

        dev = tvm.cuda()
        module = graph_executor.GraphModule(lib["default"](dev))
        return module, compile_time

    def benchmark(self, precision, num_runs=100):
        """특정 정밀도로 벤치마크"""
        if precision not in self.models:
            raise ValueError(f"Unknown precision: {precision}")

        module = self.models[precision]
        dev = tvm.cuda()

        # 입력 데이터
        dtype = "float16" if precision == "fp16" else "float32"
        input_data = np.random.randn(1, 3, 640, 640).astype("float32")
        module.set_input("images", input_data)

        # 워밍업
        for _ in range(20):
            module.run()
        dev.sync()

        # 벤치마크
        times = []
        for _ in range(num_runs):
            start = time.time()
            module.run()
            dev.sync()
            times.append((time.time() - start) * 1000)

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000 / np.mean(times)
        }


def simulate_adaptive_precision():
    """Adaptive Precision 시뮬레이션"""
    print("\n[Adaptive Precision Simulation]")
    print("=" * 50)

    # 복잡도 시나리오
    scenarios = [
        {"name": "Empty road", "complexity": 0.1, "objects": 0},
        {"name": "Light traffic", "complexity": 0.3, "objects": 3},
        {"name": "Moderate traffic", "complexity": 0.5, "objects": 8},
        {"name": "Heavy traffic", "complexity": 0.7, "objects": 15},
        {"name": "Complex intersection", "complexity": 0.9, "objects": 25},
    ]

    # 정밀도별 예상 성능 (시뮬레이션)
    precision_specs = {
        "int8": {"latency": 6, "power": 8, "accuracy": 0.92},
        "fp16": {"latency": 8, "power": 12, "accuracy": 0.97},
        "fp32": {"latency": 12, "power": 18, "accuracy": 1.0},
    }

    print("\nScenario Analysis:")
    print("-" * 70)
    print(f"{'Scenario':<25} {'Complexity':>10} {'Precision':>10} "
          f"{'Latency':>10} {'Power':>10}")
    print("-" * 70)

    for scenario in scenarios:
        # 복잡도에 따른 정밀도 선택
        if scenario["complexity"] < 0.3:
            precision = "int8"
        elif scenario["complexity"] < 0.7:
            precision = "fp16"
        else:
            precision = "fp32"

        specs = precision_specs[precision]
        print(f"{scenario['name']:<25} {scenario['complexity']:>10.1f} "
              f"{precision:>10} {specs['latency']:>8} ms {specs['power']:>8} W")

    print("-" * 70)

    # 평균 계산
    print("\nAdaptive vs Fixed Precision:")
    print("-" * 50)

    # 균등 분포 가정
    avg_latency_adaptive = (6 + 8 + 8 + 12 + 12) / 5
    avg_latency_fixed_fp32 = 12
    avg_latency_fixed_fp16 = 8

    avg_power_adaptive = (8 + 12 + 12 + 18 + 18) / 5
    avg_power_fixed_fp32 = 18
    avg_power_fixed_fp16 = 12

    print(f"  {'Method':<20} {'Avg Latency':>15} {'Avg Power':>15}")
    print(f"  {'Fixed FP32':<20} {avg_latency_fixed_fp32:>12} ms {avg_power_fixed_fp32:>12} W")
    print(f"  {'Fixed FP16':<20} {avg_latency_fixed_fp16:>12} ms {avg_power_fixed_fp16:>12} W")
    print(f"  {'Adaptive':<20} {avg_latency_adaptive:>12.1f} ms {avg_power_adaptive:>12.1f} W")

    print(f"\n  Latency reduction: {(1 - avg_latency_adaptive/avg_latency_fixed_fp32)*100:.1f}%")
    print(f"  Power reduction: {(1 - avg_power_adaptive/avg_power_fixed_fp32)*100:.1f}%")


def complexity_predictor_concept():
    """복잡도 예측기 개념 설명"""
    print("\n[Complexity Predictor Design]")
    print("=" * 50)

    print("""
복잡도 예측기 설계:

1. 입력 특징 (< 1ms로 계산 가능해야 함)
   - 이전 프레임 객체 수
   - 다운샘플된 이미지(64x64)의 엣지 밀도
   - 이전 프레임과의 차이 (움직임)
   - 시간대 정보 (optional)

2. 예측 모델
   - 간단한 decision tree 또는 linear model
   - 또는 tiny CNN (< 0.5ms)

3. 출력
   - 복잡도 점수: 0.0 ~ 1.0
   - 임계값 기반 정밀도 선택:
     * < 0.3: INT8
     * 0.3 ~ 0.7: FP16
     * > 0.7: FP32

4. 구현 고려사항
   - 예측 오버헤드가 절약보다 작아야 함
   - 연속 프레임 간 급격한 전환 방지 (hysteresis)
   - 잘못된 예측의 영향 (FP32가 필요한데 INT8 사용) 최소화
""")


def main():
    print("=" * 50)
    print("Adaptive Precision Experiment")
    print("=" * 50)

    # 모델이 있으면 실제 벤치마크
    model_path = "models/yolov5s.onnx"
    if os.path.exists(model_path):
        print("\nFound YOLO model, running actual benchmark...")
        model = MultiPrecisionModel(model_path)

        print("\n[Precision Comparison]")
        print("-" * 50)
        for precision in ["fp32", "fp16", "int8"]:
            results = model.benchmark(precision)
            print(f"  {precision.upper()}:")
            print(f"    Latency: {results['mean_ms']:.2f} ms (±{results['std_ms']:.2f})")
            print(f"    FPS: {results['fps']:.1f}")
            print(f"    Compile time: {model.compile_times[precision]:.1f} sec")
    else:
        print("\nNo YOLO model found. Running simulation...")
        print("  To run actual benchmark:")
        print("  python3 experiments/00_tvm_setup/03_compile_yolo.py")

    # Adaptive Precision 시뮬레이션
    simulate_adaptive_precision()

    # 복잡도 예측기 설계
    complexity_predictor_concept()

    print("\n" + "=" * 50)
    print("Next Steps")
    print("=" * 50)
    print("  1. 실제 데이터셋으로 정확도 측정")
    print("  2. 복잡도 예측기 구현 및 학습")
    print("  3. 런타임 모델 전환 메커니즘 구현")


if __name__ == "__main__":
    main()
