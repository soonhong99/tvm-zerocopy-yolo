# Phase 0: TVM 환경 구축

## 목표
Jetson Xavier에서 TVM 빌드 및 기본 동작 확인

## 실험 순서

### 1. TVM 설치
```bash
cd ~/2025_Thesis/nvm_practice
chmod +x scripts/01_install_tvm.sh
./scripts/01_install_tvm.sh
```

### 2. 설치 확인
```bash
python3 experiments/00_tvm_setup/01_verify_install.py
```

### 3. ResNet18 컴파일 테스트
```bash
python3 experiments/00_tvm_setup/02_compile_resnet.py
```

### 4. YOLOv5 컴파일
```bash
pip3 install ultralytics onnx onnxruntime
python3 experiments/00_tvm_setup/03_compile_yolo.py
```

## 예상 결과
- TVM 버전 확인
- CUDA 활성화 확인
- ResNet18: ~5ms 추론
- YOLOv5s: ~10-15ms 추론

## 트러블슈팅

### LLVM 관련 에러
```bash
sudo apt-get install llvm-10 llvm-10-dev
# config.cmake에서 USE_LLVM 경로 확인
```

### CUDA 에러
```bash
# JetPack CUDA 경로 확인
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 메모리 부족
```bash
# 빌드 시 병렬 수 줄이기
make -j4  # 6 대신 4
```
