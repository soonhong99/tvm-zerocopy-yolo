# TVM 환경 구축 및 YOLO 검증 완료

**날짜**: 2025년 12월 23일
**단계**: Phase 0 - TVM 환경 구축
**상태**: ✅ 완료

---

## 📋 목차

1. [개요](#개요)
2. [사전 상황](#사전-상황)
3. [진행 과정](#진행-과정)
4. [최종 결과](#최종-결과)
5. [다음 단계](#다음-단계)

---

## 개요

Jetson Xavier에서 Apache TVM을 활용한 Zero-Copy YOLO 파이프라인 연구를 위해 TVM 환경을 구축하고, YOLO 모델이 TVM에서 정상적으로 동작하는지 검증하였습니다.

### 목표
- TVM 설치 및 기본 동작 확인
- 간단한 모델(ResNet18)로 TVM 컴파일 테스트
- YOLO 모델 TVM 컴파일 및 실행 검증

---

## 사전 상황

### 발생한 문제
TVM 빌드 후 다음과 같은 import 오류 발생:
```
ImportError: cannot import name '_ffi_api' from 'tvm._ffi'
```

**원인 분석**:
1. TVM 버전: v0.23.dev0 (불안정한 개발 버전)
2. `core.so` (C++ 확장 모듈) 빌드 실패
3. Python 바인딩이 제대로 연결되지 않음

### 해결 방법
TVM 재빌드 수행:
- 실행 스크립트: `scripts/01_install_tvm.sh`
- 빌드 설정: CUDA, cuDNN, LLVM-10 활성화
- 빌드 시간: 약 1-2시간 (ninja -j6)

---

## 진행 과정

### 1단계: TVM 설치 검증

#### 실행 파일
```bash
python3 experiments/00_tvm_setup/01_verify_install.py
```

#### 검증 내용

**[1/5] TVM Import Check**
- TVM 버전: 0.18.0
- Import: ✅ PASSED

**[2/5] CUDA Check**
- CUDA Available: ✅ True
- GPU Memory Test: ✅ PASSED
- Array Shape: (1000, 1000)

**[3/5] Simple Compute Test**
- Vector Add Compile: ✅ PASSED
- Numerical Check: ✅ PASSED

**[4/5] Relay IR Test**
- Relay Compile: ✅ PASSED
- Output Shape: (1, 64, 112, 112)

**[5/5] Matrix Multiplication Benchmark**
- Matrix Size: 1024x1024 @ 1024x1024
- Average Time: 37.10 ms
- Performance: **57.9 GFLOPS**

#### 결과
```
✅ 모든 테스트 통과
TVM이 정상적으로 설치되어 Xavier GPU에서 동작 확인
```

---

### 2단계: ResNet18 컴파일 테스트

#### 실행 파일
```bash
python3 experiments/00_tvm_setup/02_compile_resnet.py
```

#### 진행 과정

**모델 준비**
- PyTorch ResNet18 pretrained 모델 다운로드
- 경로: `/home/malibu/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
- 크기: 44.7 MB

**Relay IR 변환**
- Input Shape: [1, 3, 224, 224]
- Total IR length: 23,091 chars
- Num params: 102

**컴파일**
- Target: `cuda -arch=sm_72` (Jetson Xavier)
- Opt Level: 3
- Compile Time: **276.2 sec** (~4.6분)

**벤치마킹** (100 runs)
- Output Shape: (1, 1000)
- 워밍업: 20회
- 측정: 100회

#### 성능 결과

| Metric | Value |
|--------|-------|
| Mean | 14.51 ms |
| Std | 5.66 ms |
| Min | 13.21 ms |
| Max | 68.75 ms |
| **FPS** | **68.9** |

#### 결과
```
✅ ResNet18 컴파일 및 실행 성공
TVM이 PyTorch 모델을 정상적으로 변환하고 Xavier GPU에서 추론 가능
```

---

### 3단계: YOLO 모델 준비

#### 문제 발생

**시도 1: Ultralytics YOLO export**
```bash
python3 experiments/00_tvm_setup/03_compile_yolo.py
```

**발생한 오류**:
```
Exit code 134
pthread_setaffinity_np failed
Assertion '__n < this->size()' failed
```

**원인**:
- onnxruntime의 thread affinity 설정 오류
- Ultralytics export 과정에서 crash 발생

#### 해결 방법

**시도 2: PyTorch Hub 직접 export**

새로운 스크립트 작성: `export_yolo_onnx.py`

```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                        pretrained=True, device='cpu')
model.cpu()
model.eval()

torch.onnx.export(
    model, dummy_input, output_path,
    opset_version=11,
    input_names=['images'],
    output_names=['output']
)
```

**추가 의존성 설치**:
```bash
pip3 install --user tqdm seaborn matplotlib opencv-python pandas
```

**핵심 해결 포인트**:
- ❌ CUDA device에서 export → device mismatch 오류
- ✅ CPU device 강제 지정 → 성공

#### 실행 및 결과

```bash
python3 experiments/00_tvm_setup/export_yolo_onnx.py
```

**생성된 파일**:
- 경로: `models/yolov5s.onnx`
- 크기: **27.6 MB**
- 형식: ONNX opset 11

#### 결과
```
✅ YOLO ONNX 모델 생성 성공
PyTorch Hub → ONNX 변환 완료
```

---

### 4단계: YOLO TVM 컴파일

#### 실행 파일

커스텀 스크립트 작성: `compile_yolo_tvm.py`

```bash
python3 experiments/00_tvm_setup/compile_yolo_tvm.py
```

#### 진행 과정

**[1/3] ONNX 모델 로드**
```python
onnx_model = onnx.load("models/yolov5s.onnx")
```

**[2/3] Relay IR 변환 및 컴파일**
- Input Shape: `{"images": [1, 3, 640, 640]}`
- Num params: 0 (weights embedded in graph)
- Target: `cuda -arch=sm_72`
- Opt Level: 3
- Compile Time: **217.7 sec** (~3.6분)

**[3/3] 벤치마킹**
- Warming up: 20회
- Measuring: 100회
- Device: CUDA (Xavier GPU)

#### 성능 결과

**추론 시간**:
| Metric | Value |
|--------|-------|
| Mean | **101.47 ms** |
| Std | 4.09 ms |
| Min | 98.82 ms |
| Max | 119.64 ms |
| **FPS** | **9.9** |

**출력 정보**:
- Num outputs: 1
- Output shape: `(1, 25200, 85)`
  - 25200 = grid cells (80×80 + 40×40 + 20×20)
  - 85 = bbox(4) + objectness(1) + classes(80)

**저장된 모델**:
- 경로: `models/yolov5s_tvm_fp32.so`
- 크기: **29.5 MB**

#### 주의 메시지
```
⚠️ One or more operators have not been tuned.
   Please tune your model for better performance.
```

→ AutoTVM tuning을 통해 성능 향상 가능

#### 결과
```
✅ YOLO TVM 컴파일 및 실행 성공
ONNX → Relay IR → CUDA 바이너리 변환 완료
Xavier GPU에서 9.9 FPS로 추론 가능
```

---

## 최종 결과

### 생성된 파일 목록

```
models/
├── yolov5s.onnx              28 MB  (ONNX 원본 모델)
└── yolov5s_tvm_fp32.so       30 MB  (TVM 컴파일 라이브러리)

experiments/00_tvm_setup/
├── 01_verify_install.py           (TVM 설치 검증)
├── 02_compile_resnet.py           (ResNet18 테스트)
├── 03_compile_yolo.py             (원본 YOLO 스크립트)
├── export_yolo_onnx.py            (커스텀 ONNX export)
└── compile_yolo_tvm.py            (커스텀 TVM 컴파일)

scripts/
└── 01_install_tvm.sh              (TVM 빌드 스크립트)
```

### 성능 요약

| 모델 | 입력 크기 | 추론 시간 | FPS |
|------|----------|----------|-----|
| ResNet18 | 224×224 | 14.51 ms | 68.9 |
| **YOLOv5s** | **640×640** | **101.47 ms** | **9.9** |

### 시스템 환경

```
Hardware: Jetson Xavier
JetPack: 5.x (R35.6.0)
CUDA Architecture: sm_72
Python: 3.8.10
TVM: 0.18.0
PyTorch: 2.1.0a0+41361538.nv23.06
```

---

## 발생한 문제 및 해결

### 문제 1: TVM import 오류

**문제**:
```python
ImportError: cannot import name '_ffi_api' from 'tvm._ffi'
```

**해결**:
- TVM 전체 재빌드 (`scripts/01_install_tvm.sh`)
- CUDA, cuDNN, LLVM 활성화 확인
- Python 패키지 재설치 (`pip3 install -e .`)

### 문제 2: YOLO ONNX export crash

**문제**:
```
Exit code 134
pthread_setaffinity_np failed
```

**해결**:
- Ultralytics export 대신 PyTorch Hub 사용
- CPU device 강제 지정으로 device mismatch 해결

### 문제 3: 의존성 누락

**문제**:
```
ModuleNotFoundError: No module named 'tqdm'
ModuleNotFoundError: No module named 'seaborn'
```

**해결**:
```bash
pip3 install --user tqdm seaborn matplotlib opencv-python pandas
```

### 문제 4: CUDA/CPU device mismatch

**문제**:
```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

**해결**:
```python
model = torch.hub.load(..., device='cpu')
model.cpu()
```

---

## 배운 점

### TVM 사용 팁

1. **빌드 설정 중요성**
   - `USE_CUDA`, `USE_CUDNN`, `USE_LLVM` 반드시 활성화
   - config.cmake 정확히 설정

2. **컴파일 시간**
   - 중간 크기 모델도 3-5분 소요
   - AutoTVM tuning 시 더 오래 걸림 예상

3. **성능 최적화 여지**
   - "operators have not been tuned" 메시지
   - AutoTVM으로 추가 최적화 가능

### ONNX Export 팁

1. **Device 일관성 유지**
   - Export 시 모든 텐서가 같은 device에 있어야 함
   - CPU로 통일하는 것이 안전

2. **의존성 관리**
   - YOLOv5는 많은 의존성 필요 (tqdm, seaborn, matplotlib 등)
   - 미리 설치 권장

---

## 다음 단계 (Phase 1)

### Phase 1: 베이스라인 측정

현재 **Phase 0 완료** → Phase 1 진행 준비 완료

#### 1. TensorRT 베이스라인 구축
- [ ] TensorRT로 YOLOv5s 컴파일
- [ ] TVM vs TensorRT 성능 비교
- [ ] 장단점 분석

**실행 위치**: `experiments/01_baseline_tensorrt/`

#### 2. End-to-End 레이턴시 측정
- [ ] 카메라 입력 → 전처리 → 추론 → 후처리 전체 파이프라인
- [ ] 각 단계별 시간 측정
- [ ] 병목 지점 식별

#### 3. 메모리 복사 지점 분석
- [ ] nsys 프로파일링
- [ ] CPU↔GPU 메모리 전송 식별
- [ ] 복사 제거 가능 지점 파악

**명령어**:
```bash
nsys profile -o profile_output python3 benchmark_script.py
```

#### 4. 전력 소모 측정
- [ ] tegrastats로 전력 모니터링
- [ ] TVM vs TensorRT 전력 비교
- [ ] 성능 대비 효율성 분석

**명령어**:
```bash
sudo tegrastats --interval 100
```

### Phase 2: Zero-Copy 구현 (예정)

- Jetson Unified Memory 학습 및 테스트
- TVM 전처리 커널 구현
- Zero-Copy 파이프라인 통합

**실행 위치**: `experiments/02_zero_copy/`

---

## 참고 자료

### TVM 공식 문서
- [TVM Documentation](https://tvm.apache.org/docs/)
- [TVM Tutorials](https://tvm.apache.org/docs/tutorial/)
- [Jetson TVM Optimization](https://github.com/apache/tvm/tree/main/apps/howto_deploy)

### 프로젝트 문서
- `CLAUDE.md` - 프로젝트 전체 가이드
- `scripts/01_install_tvm.sh` - TVM 설치 스크립트
- `experiments/00_tvm_setup/README.md` - 실험 가이드

### 유용한 명령어

**TVM 확인**:
```bash
python3 -c "import tvm; print(tvm.__version__); print('CUDA:', tvm.cuda().exist)"
```

**GPU 모니터링**:
```bash
nvidia-smi
sudo tegrastats --interval 100
```

**프로파일링**:
```bash
nsys profile -o output python3 script.py
```

---

## 체크리스트

### Phase 0: TVM 환경 구축 ✅

- [x] TVM 소스 빌드 (CUDA, cuDNN, LLVM 활성화)
- [x] 기본 동작 확인 (ResNet18 컴파일 및 실행)
- [x] YOLO ONNX 모델 준비
- [x] YOLO TVM 컴파일 및 실행
- [x] 성능 측정 환경 구축

### 다음 우선순위 작업

1. **TensorRT 베이스라인 측정** - TVM과 비교 기준 마련
2. **메모리 프로파일링** - Zero-Copy 최적화 대상 파악
3. **전처리/후처리 분석** - GPU 커널 구현 범위 결정

---

## 결론

✅ **TVM 환경 구축 및 YOLO 검증 완료**

- TVM이 Jetson Xavier에서 정상 동작
- YOLO 모델이 TVM으로 컴파일 가능
- 현재 성능: **9.9 FPS** (FP32, 최적화 전)

**다음 목표**:
- TensorRT 베이스라인과 비교
- Zero-Copy 파이프라인으로 성능 향상
- 메모리 복사 제거로 레이턴시 감소

---

**작성자**: Claude Code
**날짜**: 2025-12-23
**프로젝트**: TVM Zero-Copy YOLO Pipeline Research
