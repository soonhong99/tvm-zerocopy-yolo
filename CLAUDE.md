# TVM Zero-Copy YOLO Pipeline Research Project

## Project Overview

Jetson Xavier에서 YOLO를 저전력, 고효율로 실행하기 위한 Apache TVM 기반 최적화 연구 프로젝트.

**핵심 연구 목표:**
1. **Zero-Copy Pipeline**: 카메라 → 전처리 → 추론 → 후처리 전 과정을 GPU에서 메모리 복사 없이 실행
2. **Adaptive Precision**: 장면 복잡도에 따른 INT8/FP16 동적 정밀도 전환
3. **End-to-End GPU Processing**: 전처리/후처리를 TVM 커널로 직접 구현

---

## Environment

- **Hardware**: Jetson Xavier (JetPack 5.x, R35.6.0)
- **CUDA Architecture**: sm_72
- **Python**: 3.8.10
- **Target Model**: YOLOv5s/YOLOv8n

---

## Directory Structure

```
nvm_practice/
├── CLAUDE.md                    # 이 파일 - 프로젝트 가이드
├── src/
│   ├── tvm_kernels/            # TVM 커스텀 커널 (전처리/후처리)
│   ├── pipeline/               # 통합 파이프라인 코드
│   ├── models/                 # 모델 변환 및 컴파일
│   └── utils/                  # 유틸리티 함수
├── experiments/
│   ├── 00_tvm_setup/           # TVM 설치 및 기본 테스트
│   ├── 01_baseline_tensorrt/   # TensorRT 베이스라인
│   ├── 02_zero_copy/           # Zero-Copy 실험
│   ├── 03_adaptive_precision/  # Adaptive Precision 실험
│   └── 04_full_pipeline/       # 전체 파이프라인 통합
├── benchmarks/                 # 성능 측정 결과
├── configs/                    # 설정 파일
├── scripts/                    # 설치/실행 스크립트
├── tests/                      # 테스트 코드
└── docs/                       # 문서
```

---

## Research Phases

### Phase 0: TVM 환경 구축 (현재 단계)
- [ ] TVM 소스 빌드 (CUDA, cuDNN, LLVM 활성화)
- [ ] 기본 동작 확인 (ResNet18 컴파일 및 실행)
- [ ] 성능 측정 환경 구축

### Phase 1: 베이스라인 측정
- [ ] TensorRT YOLO 베이스라인 구축
- [ ] End-to-End 레이턴시 측정
- [ ] 메모리 복사 지점 식별 (nsys 프로파일링)
- [ ] 전력 소모 측정 (tegrastats)

### Phase 2: Zero-Copy 구현
- [ ] Jetson Unified Memory 학습 및 테스트
- [ ] TVM 전처리 커널 구현 (Resize, BGR→RGB, Normalize)
- [ ] TVM 후처리 커널 구현 (NMS)
- [ ] Zero-Copy 파이프라인 통합

### Phase 3: Adaptive Precision
- [ ] INT8/FP16 양자화 모델 준비
- [ ] 복잡도 예측기 설계
- [ ] 런타임 모델 전환 메커니즘
- [ ] 정확도-성능 트레이드오프 분석

### Phase 4: 최종 통합 및 평가
- [ ] 전체 파이프라인 통합
- [ ] 종합 벤치마크
- [ ] 논문 작성

---

## TVM Quick Reference

### TVM 설치 (Jetson Xavier)

```bash
# 의존성 설치
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools \
    gcc libtinfo-dev zlib1g-dev build-essential cmake \
    libedit-dev libxml2-dev llvm-10 llvm-10-dev

# TVM 소스 클론
cd ~
git clone --recursive https://github.com/apache/tvm tvm
cd tvm

# 빌드 설정
mkdir build && cp cmake/config.cmake build/
cd build

# config.cmake 수정 필수:
# set(USE_CUDA ON)
# set(USE_CUDNN ON)
# set(USE_LLVM /usr/bin/llvm-config-10)
# set(USE_GRAPH_EXECUTOR ON)
# set(USE_PROFILER ON)

cmake .. && make -j6

# Python 설치
cd ../python && pip3 install -e .
pip3 install numpy decorator attrs tornado psutil xgboost cloudpickle
```

### TVM 기본 사용법

```python
import tvm
from tvm import relay

# 1. PyTorch → Relay IR 변환
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# 2. 타겟 지정 (Xavier = sm_72)
target = tvm.target.Target("cuda -arch=sm_72")

# 3. 컴파일
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# 4. 실행
from tvm.contrib import graph_executor
module = graph_executor.GraphModule(lib["default"](tvm.cuda()))
module.set_input("input", data)
module.run()
output = module.get_output(0).numpy()
```

### TVM 커스텀 커널 작성

```python
from tvm import te

# 텐서 연산 정의
A = te.placeholder((M, N), name="A")
B = te.compute((M, N), lambda i, j: A[i, j] * 2, name="B")

# 스케줄 생성 및 GPU 매핑
s = te.create_schedule(B.op)
bx, tx = s[B].split(s[B].op.axis[0], factor=32)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))

# 빌드
func = tvm.build(s, [A, B], target="cuda -arch=sm_72")
```

---

## Key Commands

```bash
# TVM 설치 확인
python3 -c "import tvm; print(tvm.__version__); print('CUDA:', tvm.cuda().exist)"

# 전력 모니터링
sudo tegrastats --interval 100

# GPU 프로파일링
nsys profile -o output python3 script.py

# 메모리 사용량 확인
nvidia-smi

# 실험 실행
python3 experiments/00_tvm_setup/01_verify_install.py
```

---

## Research Notes

### TVM vs TensorRT 차이점

| 항목 | TensorRT | TVM |
|------|----------|-----|
| 제어 수준 | 블랙박스 | 완전 제어 가능 |
| 전처리/후처리 | 별도 구현 필요 | 통합 커널 작성 가능 |
| 메모리 관리 | 자동 | 직접 제어 가능 |
| 커스텀 최적화 | 불가능 | 스케줄 직접 작성 |
| 코드 확인 | 불가능 | 생성된 CUDA 확인 가능 |

### Zero-Copy 핵심 아이디어

```
기존 (TensorRT):
카메라 → [복사] → CPU 전처리 → [복사] → GPU 추론 → [복사] → CPU 후처리
                    ↑ 병목!                        ↑ 병목!

목표 (TVM Zero-Copy):
카메라 → Unified Memory → GPU 전처리 → GPU 추론 → GPU 후처리
         ↑ 복사 없음!      ↑ TVM 커널    ↑ TVM 커널
```

### Adaptive Precision 전략

- 복잡도 낮음 (단순 배경): INT8 사용 → 빠름, 저전력
- 복잡도 높음 (복잡한 장면): FP16 사용 → 정확함
- 전환 오버헤드 최소화가 핵심

---

## Coding Guidelines

1. **실험 재현성**: 모든 실험은 seed 고정, 결과 저장
2. **측정 정확성**: GPU 동기화 후 시간 측정, 워밍업 필수
3. **문서화**: 각 실험 디렉토리에 README.md 작성
4. **버전 관리**: 주요 변경사항은 git commit

---

## Useful Resources

- [TVM 공식 문서](https://tvm.apache.org/docs/)
- [TVM 튜토리얼](https://tvm.apache.org/docs/tutorial/)
- [Jetson TVM 최적화](https://github.com/apache/tvm/tree/main/apps/howto_deploy)
- [YOLO TVM 예제](https://tvm.apache.org/docs/how_to/compile_models/from_pytorch.html)

---

## Current Status

**진행 상태**: Phase 0 - TVM 환경 구축 시작 전
**다음 할 일**: TVM 소스 빌드 스크립트 실행
