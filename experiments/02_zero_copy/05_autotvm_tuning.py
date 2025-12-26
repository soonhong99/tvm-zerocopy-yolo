#!/usr/bin/env python3
"""
AutoTVM으로 YOLOv5s 모델 튜닝
TVM 추론 성능 최적화 (112ms → 30-40ms 목표)

실행: python3 experiments/02_zero_copy/05_autotvm_tuning.py

주의사항:
- 튜닝에 수 시간 소요 (trials 수에 따라 다름)
- GPU 사용률 100%로 발열 주의
- 중간에 중단되어도 로그는 저장됨 (이어서 가능)
"""

import os
import time
import numpy as np
import onnx

import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib import graph_executor


class AutoTVMTuner:
    """YOLOv5s AutoTVM 튜닝"""

    def __init__(self,
                 model_path="models/yolov5s.onnx",
                 log_file="models/yolov5s_autotvm.log",
                 tuned_model_path="models/yolov5s_tvm_tuned.so"):
        """
        Args:
            model_path: ONNX 모델 경로
            log_file: AutoTVM 튜닝 로그 저장 경로
            tuned_model_path: 튜닝된 TVM 모델 저장 경로
        """
        self.model_path = model_path
        self.log_file = log_file
        self.tuned_model_path = tuned_model_path

        # Jetson Xavier CUDA 타겟
        self.target = tvm.target.Target("cuda -arch=sm_72")
        self.device = tvm.cuda()

        self.mod = None
        self.params = None
        self.input_name = None
        self.input_shape = None

    def load_model(self):
        """ONNX 모델을 Relay IR로 변환"""
        print("\n[1/5] Loading ONNX Model")
        print("=" * 60)

        onnx_model = onnx.load(self.model_path)

        # YOLO 입력 shape
        self.input_shape = (1, 3, 640, 640)
        self.input_name = "images"

        shape_dict = {self.input_name: self.input_shape}

        # ONNX → Relay IR
        self.mod, self.params = relay.frontend.from_onnx(onnx_model, shape_dict)

        print(f"  Model: {self.model_path}")
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Relay IR converted successfully")

    def extract_tasks(self):
        """튜닝할 연산자(task) 추출"""
        print("\n[2/5] Extracting Tunable Tasks")
        print("=" * 60)

        tasks = autotvm.task.extract_from_program(
            self.mod["main"],
            target=self.target,
            params=self.params
        )

        print(f"  Found {len(tasks)} tunable tasks:")
        for i, task in enumerate(tasks):
            print(f"    [{i+1}] {task.name}: {task.flop/1e9:.2f} GFLOPS")

        return tasks

    def tune_tasks(self, tasks, n_trial=1000, tuner='xgb'):
        """
        각 task에 대해 AutoTVM 튜닝 실행

        Args:
            tasks: 튜닝할 task 리스트
            n_trial: 각 task당 시도 횟수 (많을수록 좋지만 오래 걸림)
            tuner: 'xgb', 'ga', 'random', 'gridsearch'
        """
        print(f"\n[3/5] AutoTVM Tuning ({tuner.upper()}, {n_trial} trials per task)")
        print("=" * 60)
        print(f"  Log file: {self.log_file}")
        print(f"  Estimated time: {len(tasks) * n_trial * 0.5 / 60:.1f} minutes")
        print("")

        # Tuner 선택
        tuner_map = {
            'xgb': XGBTuner,
            'ga': GATuner,
            'random': RandomTuner,
            'gridsearch': GridSearchTuner
        }

        if tuner not in tuner_map:
            raise ValueError(f"Unknown tuner: {tuner}")

        # 기존 로그 있으면 이어서 튜닝
        if os.path.exists(self.log_file):
            print(f"  ⚠️ Found existing log: {self.log_file}")
            print(f"  Continuing from previous tuning...")

        # 각 task 튜닝
        for i, task in enumerate(tasks):
            prefix = f"[{i+1}/{len(tasks)}]"

            print(f"\n{prefix} Tuning {task.name}")
            print(f"  Workload: {task.workload}")
            print(f"  Config space size: {len(task.config_space)}")

            # Tuner 생성
            tuner_obj = tuner_map[tuner](task)

            # 측정 옵션
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(
                    number=5,        # 각 config를 5번 실행
                    repeat=3,        # 3번 반복하여 중간값 사용
                    timeout=10,
                    min_repeat_ms=150
                )
            )

            # 튜닝 실행
            start_time = time.time()

            tuner_obj.tune(
                n_trial=min(n_trial, len(task.config_space)),
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
                    autotvm.callback.log_to_file(self.log_file)
                ]
            )

            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f}s")

        print("\n" + "=" * 60)
        print(f"✅ Tuning completed! Log saved to: {self.log_file}")

    def compile_with_best_config(self):
        """튜닝 결과를 사용하여 최적화된 모델 컴파일"""
        print(f"\n[4/5] Compiling with Best Configurations")
        print("=" * 60)

        if not os.path.exists(self.log_file):
            raise FileNotFoundError(f"Tuning log not found: {self.log_file}")

        # 튜닝 로그 적용
        with autotvm.apply_history_best(self.log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(self.mod, target=self.target, params=self.params)

        # 모델 저장
        lib.export_library(self.tuned_model_path)

        print(f"  Optimized model saved: {self.tuned_model_path}")
        print(f"  Size: {os.path.getsize(self.tuned_model_path) / 1024 / 1024:.1f} MB")

        return lib

    def benchmark_model(self, lib, warmup=10, repeat=100):
        """튜닝된 모델 성능 측정"""
        print(f"\n[5/5] Benchmarking Tuned Model")
        print("=" * 60)

        # Runtime 생성
        module = graph_executor.GraphModule(lib["default"](self.device))

        # 입력 데이터 준비
        input_data = np.random.randn(*self.input_shape).astype("float32")
        module.set_input(self.input_name, tvm.nd.array(input_data, self.device))

        # 워밍업
        for _ in range(warmup):
            module.run()
        self.device.sync()

        # 벤치마크
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            module.run()
            self.device.sync()
            times.append((time.perf_counter() - start) * 1000)

        mean_time = np.mean(times)
        std_time = np.std(times)
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)

        print(f"  Mean:   {mean_time:.2f} ms (±{std_time:.2f})")
        print(f"  Median: {p50:.2f} ms")
        print(f"  P95:    {p95:.2f} ms")
        print(f"  FPS:    {1000/mean_time:.1f}")

        return mean_time

    def compare_with_baseline(self, tuned_time):
        """베이스라인과 비교"""
        print("\n" + "=" * 60)
        print("Performance Comparison")
        print("=" * 60)

        # 이전 결과 (benchmarks/trt_vs_tvm_comparison.txt에서)
        baseline_tvm = 112.56  # ms
        tensorrt_fp16 = 29.27  # ms

        speedup_vs_baseline = baseline_tvm / tuned_time
        vs_tensorrt = tuned_time / tensorrt_fp16

        print(f"  TVM FP32 (Before Tuning):  {baseline_tvm:.2f} ms")
        print(f"  TVM FP32 (After Tuning):   {tuned_time:.2f} ms")
        print(f"  Speedup:                   {speedup_vs_baseline:.2f}x")
        print(f"  Improvement:               {baseline_tvm - tuned_time:.2f} ms")
        print("")
        print(f"  TensorRT FP16:             {tensorrt_fp16:.2f} ms")
        print(f"  TVM vs TensorRT:           {vs_tensorrt:.2f}x")

        if tuned_time < tensorrt_fp16:
            print(f"  🎉 TVM is FASTER than TensorRT FP16!")
        elif tuned_time < tensorrt_fp16 * 1.5:
            print(f"  ✅ TVM is competitive with TensorRT FP16")
        else:
            print(f"  ⚠️  TVM is still slower than TensorRT FP16")

        # 결과 저장
        result_file = "benchmarks/autotvm_results.txt"
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

        with open(result_file, "w") as f:
            f.write("AutoTVM Tuning Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: YOLOv5s\n")
            f.write(f"Target: {self.target}\n")
            f.write(f"Tuning log: {self.log_file}\n\n")
            f.write(f"Before Tuning: {baseline_tvm:.2f} ms\n")
            f.write(f"After Tuning:  {tuned_time:.2f} ms\n")
            f.write(f"Speedup:       {speedup_vs_baseline:.2f}x\n")
            f.write(f"Improvement:   {baseline_tvm - tuned_time:.2f} ms\n\n")
            f.write(f"vs TensorRT FP16: {vs_tensorrt:.2f}x\n")

        print(f"\n  Results saved to: {result_file}")


def quick_tuning():
    """빠른 튜닝 (테스트용, 낮은 trial 수)"""
    print("=" * 60)
    print("AutoTVM Quick Tuning (Test Mode)")
    print("=" * 60)
    print("\n⚠️  WARNING: This is a QUICK test with low trial count")
    print("    For production, use full_tuning() with n_trial=2000+")

    tuner = AutoTVMTuner()
    tuner.load_model()
    tasks = tuner.extract_tasks()

    # 빠른 테스트: 각 task당 100 trials
    tuner.tune_tasks(tasks, n_trial=100, tuner='xgb')

    lib = tuner.compile_with_best_config()
    tuned_time = tuner.benchmark_model(lib)
    tuner.compare_with_baseline(tuned_time)


def full_tuning(n_trial=1500):
    """전체 튜닝 (프로덕션용, 높은 trial 수)"""
    print("=" * 60)
    print(f"AutoTVM Full Tuning ({n_trial} trials per task)")
    print("=" * 60)
    print("\n⏱️  Estimated time: Several hours")
    print("   You can safely interrupt (Ctrl+C) and resume later")

    tuner = AutoTVMTuner()
    tuner.load_model()
    tasks = tuner.extract_tasks()

    # 전체 튜닝
    tuner.tune_tasks(tasks, n_trial=n_trial, tuner='xgb')

    lib = tuner.compile_with_best_config()
    tuned_time = tuner.benchmark_model(lib)
    tuner.compare_with_baseline(tuned_time)


def compile_from_existing_log():
    """이미 있는 튜닝 로그로 모델만 컴파일"""
    print("=" * 60)
    print("Compile from Existing Tuning Log")
    print("=" * 60)

    tuner = AutoTVMTuner()
    tuner.load_model()

    lib = tuner.compile_with_best_config()
    tuned_time = tuner.benchmark_model(lib)
    tuner.compare_with_baseline(tuned_time)


def main():
    import sys

    print("\n" + "=" * 60)
    print("AutoTVM Tuning for YOLOv5s")
    print("=" * 60)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nUsage:")
        print("  python3 05_autotvm_tuning.py [mode]")
        print("\nModes:")
        print("  quick     - Quick test (100 trials, ~30 min)")
        print("  full      - Full tuning (1500 trials, ~3-5 hours)")
        print("  compile   - Compile from existing log (no tuning)")
        print("\nDefault: quick")
        mode = "quick"

    if mode == "quick":
        quick_tuning()
    elif mode == "full":
        full_tuning(n_trial=1500)
    elif mode == "compile":
        compile_from_existing_log()
    else:
        print(f"Unknown mode: {mode}")
        return

    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("  1. If you ran 'quick', consider running 'full' for better results")
    print("  2. Implement E2E pipeline (04_e2e_pipeline.py)")
    print("  3. Compare TensorRT vs TVM Zero-Copy end-to-end")


if __name__ == "__main__":
    main()
