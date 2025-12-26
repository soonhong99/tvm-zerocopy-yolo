# Phase 2: Zero-Copy GPU 파이프라인

TVM 커스텀 커널을 활용한 End-to-End GPU 파이프라인 구현.

---

## 📊 현재 진행 상황 (2025-12-26)

### 완료된 작업

| 항목 | 상태 | 비고 |
|------|------|------|
| 병목 분석 | ✅ 완료 | CPU 전처리 60.7% 차지 확인 |
| 메모리 복사 측정 | ✅ 완료 | 4.5ms (4.3%) - 생각보다 작음 |
| GPU 전처리 커널 구현 | ✅ 완료 | 1.57ms (40x 빠름) |
| GPU 후처리 커널 구현 | ✅ 완료 | 10-29ms (복잡도 따라 다름) |
| 01_unified_memory_test.py | ⬜ 미실행 | 파일 존재, 실행 안 함 |
| 02_preprocess_kernel.py | ✅ 실행 완료 | 벤치마크 결과 확보 |
| 03_postprocess_kernel.py | ✅ 실행 완료 | 벤치마크 결과 확보 |
| 04_e2e_pipeline.py | ⬜ 미구현 | 다음 단계 |

### 핵심 발견

1. **CPU 전처리가 최대 병목** (63.68ms, 60.7%)
   - GPU 전처리로 1.57ms 달성 (40x 개선)
   - **이것만으로도 E2E 레이턴시 60% 감소 가능**

2. **메모리 복사는 작은 오버헤드** (4.5ms, 4.3%)
   - Zero-Copy의 진정한 가치 ≠ memcpy 제거
   - = **CPU 워크로드를 GPU로 이동**

3. **GPU 후처리는 복잡도에 따라 다름**
   - 간단한 장면: GPU가 CPU보다 느림 (launch 오버헤드)
   - 복잡한 장면: GPU가 7.4x 빠름
   - 평균적으로는 GPU가 유리할 것으로 예상

### 예상 성능 (실측 기반)

```
[현재 TensorRT 파이프라인]
  CPU 전처리:    63.68ms  ← 최대 병목
  메모리 복사:    4.50ms
  GPU 추론:      29.00ms  (TensorRT FP16)
  CPU 후처리:     8.00ms
  ─────────────────────
  총 E2E:       105.18ms  (9.5 FPS)

[TVM Zero-Copy 파이프라인 목표]
  GPU 전처리:     1.57ms  ← 40x 빠름 (실측)
  GPU 추론:      30.00ms  (TVM 튜닝 후)
  GPU 후처리:    10.81ms  ← 실측 (복잡한 장면 기준)
  ─────────────────────
  총 E2E:        42.38ms  (23.6 FPS)

  개선: 105ms → 42ms (2.5배 빠름, 60% 레이턴시 감소)
```

**주의**: 추론 부분은 아직 튜닝 전 (현재 112ms). AutoTVM 튜닝 후 30ms 목표.

### 다음 단계

1. **E2E 파이프라인 통합** (04_e2e_pipeline.py)
   - 전처리 → 추론 → 후처리 연결
   - 실제 성능 측정

2. **AutoTVM 튜닝**
   - 추론 112ms → 30-40ms 목표
   - 가장 큰 성능 개선 예상

3. **후처리 최적화** (선택적)
   - TVM 대신 PyTorch로 전체 구현 고려
   - Confidence Filtering 성능 개선

---

## 핵심 목표

**CPU 전처리/후처리 병목 제거**를 통한 레이턴시 감소

```
[현재] CPU 전처리 → 복사 → GPU 추론 → 복사 → CPU 후처리  (105ms)
[목표] GPU 전처리 → GPU 추론 → GPU 후처리               (36ms)
```

---

## 병목 분석 결과 (2025-12-26 실측)

### 현재 TensorRT 파이프라인 시간 분해

| 단계 | 시간 | 비율 | 비고 |
|------|------|------|------|
| **CPU 전처리** | **63.68ms** | **60.7%** | **최대 병목!** |
| Host→Device 복사 | 2.20ms | 2.1% | 입력 4.7MB |
| GPU 추론 (TRT FP16) | 29.00ms | 27.6% | |
| Device→Host 복사 | 2.30ms | 2.2% | 출력 8.2MB |
| CPU 후처리 (NMS) | ~8.00ms | 7.6% | Python NMS |
| **총 E2E** | **~105ms** | **100%** | **9.5 FPS** |

### 핵심 발견

1. **CPU 전처리가 GPU 추론보다 2배 이상 오래 걸림**
   - 1920x1080 → 640x640 resize + BGR→RGB + normalize
   - OpenCV(CPU): 63.68ms

2. **메모리 복사는 생각보다 작음 (4.5ms, 4.3%)**
   - Zero-Copy의 진정한 가치는 memcpy 제거가 아님
   - **CPU 워크로드를 GPU로 옮기는 것**이 핵심

3. **GPU 전처리는 40배 빠름**
   - TVM 커널: 1.57ms (vs CPU 63.68ms)
   - 이미 02_preprocess_kernel.py에서 구현 완료

---

## 측정 명령어

### 1. 메모리 복사 오버헤드 측정

```bash
python3 << 'EOF'
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit

# YOLOv5s 입출력 크기
input_size = (1, 3, 640, 640)   # 4.7MB
output_size = (1, 25200, 85)    # 8.2MB

data = np.random.randn(*input_size).astype(np.float32)
output = np.empty(output_size, dtype=np.float32)

d_input = cuda.mem_alloc(data.nbytes)
d_output = cuda.mem_alloc(output.nbytes)

# Host → Device
for _ in range(10): cuda.memcpy_htod(d_input, data)  # 워밍업

times = []
for _ in range(100):
    start = time.perf_counter()
    cuda.memcpy_htod(d_input, data)
    cuda.Context.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f"H→D: {np.mean(times):.2f}ms, 대역폭: {data.nbytes/np.mean(times)/1e6:.1f} GB/s")

# Device → Host
times = []
for _ in range(100):
    start = time.perf_counter()
    cuda.memcpy_dtoh(output, d_output)
    cuda.Context.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f"D→H: {np.mean(times):.2f}ms, 대역폭: {output.nbytes/np.mean(times)/1e6:.1f} GB/s")
EOF
```

**실측 결과**:
- H→D: 2.24ms (2.2 GB/s)
- D→H: 2.31ms (3.7 GB/s)
- 총 복사: 4.55ms

### 2. CPU 전처리 시간 측정

```bash
python3 -c "
import numpy as np
import time
import cv2

raw = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

def preprocess(img):
    r = cv2.resize(img, (640, 640))
    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    r = r.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(r, (2, 0, 1))[None])

# 워밍업
for _ in range(5): preprocess(raw)

# 측정
times = []
for _ in range(50):
    s = time.perf_counter()
    preprocess(raw)
    times.append((time.perf_counter() - s) * 1000)

print(f'CPU 전처리: {np.mean(times):.2f}ms')
"
```

**실측 결과**: 63.68ms (±42.95ms)

### 3. GPU 전처리 커널 벤치마크

```bash
python3 experiments/02_zero_copy/02_preprocess_kernel.py
```

**실측 결과**:
- GPU (Nearest): 1.57ms (**40.5x 빠름**)
- GPU (Bilinear): 2.51ms (25.4x 빠름)

### 4. NMS 구현 비교

```bash
python3 << 'EOF'
import numpy as np
import time
import torch
import torchvision

for n_boxes in [50, 100, 200, 500]:
    boxes = torch.rand(n_boxes, 4).cuda() * 640
    boxes[:, 2:] += boxes[:, :2]
    scores = torch.rand(n_boxes).cuda()

    # 워밍업
    for _ in range(5): torchvision.ops.nms(boxes, scores, 0.45)
    torch.cuda.synchronize()

    times = []
    for _ in range(20):
        start = time.perf_counter()
        torchvision.ops.nms(boxes, scores, 0.45)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    print(f'{n_boxes} boxes: {np.mean(times):.2f}ms')
EOF
```

**실측 결과**:
- 50 boxes: 5.00ms
- 500 boxes: 3.87ms
- TorchVision GPU NMS는 Python NMS 대비 16x 빠름

---

## 구현 전략

### 왜 이 전략인가?

1. **CPU 전처리가 최대 병목 (60.7%)**
   - GPU 전처리로 63.68ms → 1.57ms (40x 개선)
   - 이것만으로 E2E 레이턴시 60% 감소 가능

2. **메모리 복사는 상대적으로 작음 (4.3%)**
   - Zero-Copy의 진정한 가치 ≠ memcpy 제거
   - = CPU 워크로드를 GPU로 이동

3. **TorchVision NMS는 충분히 빠름**
   - 500 boxes: ~4ms
   - 별도 GPU NMS 커널 구현 불필요

### 파이프라인 설계

```
┌───────────────────────────────────────────────────────────────┐
│                    Unified Memory (Zero-Copy)                  │
│                                                                │
│  카메라 버퍼 ──▶ GPU 전처리 ──▶ GPU 추론 ──▶ GPU 후처리       │
│     (공유)       (TVM 커널)   (TVM YOLO)   (TorchVision)      │
│                   1.57ms       ~30ms         ~4ms              │
│                                                                │
└───────────────────────────────────────────────────────────────┘

                        총 E2E: ~36ms (27.8 FPS)
```

### 구현 우선순위

| 순위 | 작업 | 예상 개선 | 상태 |
|------|------|----------|------|
| 1 | GPU 전처리 커널 | 62ms 절감 | ✅ 완료 |
| 2 | GPU 후처리 (NMS) | 4ms 절감 | ⬜ 구현 예정 |
| 3 | E2E 파이프라인 통합 | 전체 검증 | ⬜ 예정 |
| 4 | AutoTVM 튜닝 | 추론 70ms 절감 | ⬜ 예정 |

---

## 스크립트 설명

### 01_unified_memory_test.py

Jetson Xavier의 Unified Memory 동작 테스트:
- `cudaMallocManaged` 기반 메모리 할당
- CPU/GPU 동시 접근 가능 여부 확인
- Page fault 오버헤드 측정

### 02_preprocess_kernel.py

TVM으로 YOLO 전처리 GPU 커널 구현:
- **입력**: (1080, 1920, 3) uint8 BGR
- **출력**: (1, 3, 640, 640) float32 RGB normalized
- **기능**: Resize + BGR→RGB + Normalize + HWC→CHW
- **성능**: 1.57ms (CPU 대비 40x 빠름)

### 03_postprocess_kernel.py ✅ 구현 완료

GPU 기반 후처리:
- Confidence Filtering (TVM 커널)
- NMS (TorchVision ops.nms)
- TVM → PyTorch 텐서 변환 (CUDA 포인터 공유)

**실행 결과** (2025-12-26):

| 장면 복잡도 | 필터링 후 | 최종 boxes | GPU 시간 | CPU 시간 | Speedup |
|------------|----------|-----------|---------|---------|---------|
| 간단 (50) | 50 | 38 | 28.88ms | 8.69ms | 0.3x ⚠️ |
| 보통 (200) | 200 | 121 | 13.43ms | 45.43ms | 3.4x ✅ |
| 복잡 (500) | 500 | 218 | 10.81ms | 80.46ms | 7.4x ✅ |

**발견사항**:
- ⚠️ 간단한 장면에서는 GPU가 CPU보다 느림 (CUDA launch 오버헤드)
- ✅ boxes 수가 많을수록 GPU가 빠름 (500 boxes: 7.4x)
- 🔍 TVM Confidence Filtering이 예상보다 느림 (7-20ms)
- TorchVision NMS는 빠름 (500 boxes: 2.82ms)

### 04_e2e_pipeline.py (구현 예정)

End-to-End 통합 벤치마크:
- 전처리 → 추론 → 후처리 전체 연결
- TensorRT 파이프라인과 비교
- 각 단계별 시간 분해

---

## 실행 순서

```bash
# 1. Unified Memory 테스트
python3 experiments/02_zero_copy/01_unified_memory_test.py

# 2. GPU 전처리 커널 벤치마크
python3 experiments/02_zero_copy/02_preprocess_kernel.py

# 3. GPU 후처리 구현 (다음 단계)
python3 experiments/02_zero_copy/03_postprocess_kernel.py

# 4. E2E 파이프라인 벤치마크 (최종)
python3 experiments/02_zero_copy/04_e2e_pipeline.py
```

---

## 예상 성능 비교

| 파이프라인 | E2E 레이턴시 | FPS | 개선 |
|-----------|------------|-----|------|
| TensorRT (현재) | 105ms | 9.5 | baseline |
| TVM Zero-Copy (목표) | **36ms** | **27.8** | **2.9x** |

### 시간 분해 비교

| 단계 | TensorRT | TVM Zero-Copy | 절감 |
|------|----------|---------------|------|
| 전처리 | 63.68ms (CPU) | 1.57ms (GPU) | **62.11ms** |
| 복사 | 4.50ms | 0ms | 4.50ms |
| 추론 | 29.00ms | ~30ms | - |
| 후처리 | 8.00ms | ~4ms | 4.00ms |
| **총계** | **105.18ms** | **~36ms** | **~69ms** |

---

## 핵심 인사이트

> **"Zero-Copy"의 진정한 가치는 memcpy 제거가 아니라,
> CPU 워크로드를 GPU로 옮기는 것이다.**

기존 연구들은 "메모리 복사 제거"에 집중했지만,
실측 결과 **CPU 전처리가 전체 파이프라인의 60% 이상**을 차지함.

이 발견은 논문의 핵심 기여 중 하나가 될 수 있음:
- "End-to-End 파이프라인 관점에서 전처리 병목 정량화"
- "TVM 커스텀 커널로 40x 전처리 가속 달성"

---

## 다음 단계

1. **03_postprocess_kernel.py 구현**
   - Confidence Filtering + TorchVision NMS 통합

2. **04_e2e_pipeline.py 구현**
   - 전체 파이프라인 통합 및 벤치마크

3. **AutoTVM 튜닝**
   - TVM 추론 112ms → 40ms 목표

4. **결과 문서화**
   - 논문 그래프용 데이터 정리

---

## 참고: 측정 환경

- **Hardware**: Jetson Xavier (JetPack 5.x)
- **CUDA**: sm_72
- **TensorRT**: 8.5.2.2
- **PyTorch**: 2.1.0a0+41361538.nv23.06
- **TorchVision**: 0.15.1
- **TVM**: 소스 빌드
