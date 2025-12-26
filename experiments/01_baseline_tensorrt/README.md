# Phase 1: TensorRT 베이스라인 측정

TVM Zero-Copy 파이프라인의 성능 향상을 정량화하기 위한 TensorRT 베이스라인 구축.

## 목적

1. **정량적 비교 기준 마련**: TVM 최적화 효과를 "TensorRT 대비 X% 개선"으로 표현
2. **공정한 비교**: 동일 모델(YOLOv5s), 동일 하드웨어(Xavier), 동일 정밀도에서 비교
3. **최적화 목표 설정**: TensorRT 성능을 넘어서는 것이 연구 목표

## 실행 순서

```bash
# 1. TensorRT 엔진 빌드 (FP32, FP16)
python3 experiments/01_baseline_tensorrt/01_trt_build_engine.py

# 2. TensorRT 벤치마크
python3 experiments/01_baseline_tensorrt/02_trt_benchmark.py

# 3. TensorRT vs TVM 비교
python3 experiments/01_baseline_tensorrt/03_compare_trt_tvm.py
```

## 스크립트 설명

### 01_trt_build_engine.py

ONNX 모델을 TensorRT 엔진으로 변환:

- **입력**: `models/yolov5s.onnx`
- **출력**:
  - `models/yolov5s_trt_fp32.engine`
  - `models/yolov5s_trt_fp16.engine`
- **소요 시간**: 2-5분

### 02_trt_benchmark.py

TensorRT 엔진 성능 측정:

- 워밍업 20회 + 벤치마크 100회
- 측정 항목: Mean, Std, Min, Max, P95, P99, FPS
- **결과 저장**: `benchmarks/tensorrt_baseline.txt`

### 03_compare_trt_tvm.py

TensorRT와 TVM 성능 비교 분석:

- 동일 조건에서 두 프레임워크 비교
- 속도 차이(speedup) 계산
- 다음 단계 권장사항 제시

## 예상 결과

| 모델 | 정밀도 | 예상 레이턴시 | 예상 FPS |
|------|--------|--------------|----------|
| TensorRT | FP32 | ~40-50 ms | ~20-25 |
| TensorRT | FP16 | ~15-25 ms | ~40-65 |
| TVM (미튜닝) | FP32 | ~100 ms | ~10 |
| TVM (튜닝 후) | FP32 | ~30-40 ms | ~25-35 |

## 주의사항

### TensorRT 버전

JetPack 5.x에 포함된 TensorRT 사용:
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
# 예상: 8.5.x 또는 8.6.x
```

### pycuda 설치

TensorRT Python API에 필요:
```bash
pip3 install pycuda
```

### 메모리 부족 시

워크스페이스 크기 조정:
```python
# 01_trt_build_engine.py에서 workspace_gb 파라미터 수정
build_engine(..., workspace_gb=1.0)  # 기본값 2.0 → 1.0
```

## 결과 해석

### TensorRT가 TVM보다 빠른 경우

이는 **예상된 결과**입니다:
1. TVM은 AutoTVM 튜닝 전 상태
2. TensorRT는 NVIDIA 최적화가 기본 적용됨
3. 튜닝 후 TVM이 따라잡거나 능가 가능

### TVM의 차별화 포인트

TensorRT 대비 TVM의 장점:
1. **Zero-Copy Pipeline**: 전처리/후처리 통합으로 메모리 복사 제거
2. **커스텀 커널**: 애플리케이션 특화 최적화 가능
3. **투명성**: 생성된 CUDA 코드 확인/수정 가능
4. **이식성**: 다른 하드웨어로 확장 용이

## 다음 단계

베이스라인 측정 완료 후:

```bash
# Zero-Copy 테스트
python3 experiments/02_zero_copy/01_unified_memory_test.py

# 전처리 커널 테스트
python3 experiments/02_zero_copy/02_preprocess_kernel.py

# Adaptive Precision 테스트
python3 experiments/03_adaptive_precision/01_quantization_compare.py
```

## 문제 해결

### TensorRT import 실패

```bash
# JetPack TensorRT 경로 확인
python3 -c "import tensorrt"

# 실패 시 LD_LIBRARY_PATH 확인
echo $LD_LIBRARY_PATH
# /usr/lib/aarch64-linux-gnu 포함되어야 함
```

### pycuda import 실패

```bash
pip3 install --user pycuda

# 또는 CUDA 경로 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 엔진 빌드 실패

```bash
# 로그 확인을 위해 verbose 모드 실행
python3 -c "
from experiments.baseline_tensorrt.trt_build_engine import build_engine
build_engine('models/yolov5s.onnx', 'test.engine', verbose=True)
"
```
