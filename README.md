# TVM Zero-Copy YOLO Pipeline for Jetson Xavier

Apache TVM 기반 YOLO 최적화 연구 프로젝트. Jetson Xavier에서 저전력, 고효율 객체 탐지 파이프라인 구현.

## Research Goals

1. **Zero-Copy Pipeline**: 메모리 복사 없이 카메라 → 전처리 → 추론 → 후처리 전 과정을 GPU에서 실행
2. **Adaptive Precision**: 장면 복잡도에 따른 INT8/FP16/FP32 동적 정밀도 전환
3. **End-to-End GPU Processing**: TVM으로 전처리/후처리 커널 직접 구현

## Why TVM over TensorRT?

| Feature | TensorRT | TVM |
|---------|----------|-----|
| Control Level | Black-box | Full kernel control |
| Pre/Post Processing | Separate | Integrated GPU kernels |
| Memory Management | Automatic | Direct control (Zero-Copy) |
| Custom Optimization | Limited | Schedule-based tuning |
| Generated Code | Hidden | Inspectable CUDA code |

## Project Structure

```
├── CLAUDE.md                    # Detailed project guide
├── experiments/
│   ├── 00_tvm_setup/           # TVM installation & verification
│   ├── 01_baseline_tensorrt/   # TensorRT baseline
│   ├── 02_zero_copy/           # Zero-Copy pipeline experiments
│   ├── 03_adaptive_precision/  # Precision switching experiments
│   └── 04_full_pipeline/       # Integrated pipeline
├── src/                        # Core implementation
├── configs/                    # Configuration files
├── scripts/                    # Installation scripts
└── benchmarks/                 # Performance results
```

## Environment

- **Hardware**: NVIDIA Jetson Xavier (JetPack 5.x)
- **CUDA Architecture**: sm_72
- **Framework**: Apache TVM

## Quick Start

### 1. Install TVM
```bash
./scripts/01_install_tvm.sh
source ~/.bashrc
```

### 2. Verify Installation
```bash
python3 experiments/00_tvm_setup/01_verify_install.py
```

### 3. Run Experiments
```bash
# Step by step
python3 experiments/00_tvm_setup/02_compile_resnet.py
python3 experiments/00_tvm_setup/03_compile_yolo.py
python3 experiments/02_zero_copy/01_unified_memory_test.py
```

## Key Concepts

### Zero-Copy on Jetson

```
Traditional Pipeline (with copies):
Camera → [COPY] → CPU Preprocess → [COPY] → GPU Inference → [COPY] → CPU Postprocess

Zero-Copy Pipeline (this research):
Camera → Unified Memory → GPU Preprocess → GPU Inference → GPU Postprocess
         ↑ No copy!       ↑ TVM Kernel    ↑ TVM Kernel
```

### Adaptive Precision

- **Low complexity** (empty road): INT8 → Fast, low power
- **Medium complexity** (normal traffic): FP16 → Balanced
- **High complexity** (crowded scene): FP32 → Accurate

## Expected Results

| Metric | Baseline (TensorRT) | Optimized (TVM) |
|--------|---------------------|-----------------|
| Latency | ~15ms | ~10ms |
| Power | ~18W | ~12W |
| FPS | ~65 | ~100 |

## License

MIT License

## References

- [Apache TVM Documentation](https://tvm.apache.org/docs/)
- [Jetson Xavier Developer Guide](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit)
