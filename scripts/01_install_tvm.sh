#!/bin/bash
# TVM Installation Script for Jetson Xavier (JetPack 5.x)
# 예상 소요 시간: 1-2시간

set -e

echo "============================================"
echo "TVM Installation for Jetson Xavier"
echo "============================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Step 1: 의존성 설치
log_info "Step 1: Installing dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3 python3-dev python3-setuptools python3-pip \
    gcc g++ libtinfo-dev zlib1g-dev build-essential cmake \
    libedit-dev libxml2-dev \
    llvm-10 llvm-10-dev \
    ninja-build

# Python 패키지
pip3 install --upgrade pip
pip3 install numpy decorator attrs tornado psutil scipy \
    xgboost cloudpickle pytest

# Step 2: TVM 소스 클론
TVM_HOME="$HOME/tvm"
if [ -d "$TVM_HOME" ]; then
    log_warn "TVM directory exists. Pulling latest..."
    cd "$TVM_HOME"
    git pull
    git submodule update --init --recursive
else
    log_info "Step 2: Cloning TVM repository..."
    cd "$HOME"
    git clone --recursive https://github.com/apache/tvm tvm
fi

cd "$TVM_HOME"

# Step 3: 빌드 설정
log_info "Step 3: Configuring build..."
mkdir -p build
cd build

# config.cmake 생성
cat > config.cmake << 'EOF'
# TVM Build Configuration for Jetson Xavier

# CUDA 활성화 (Xavier GPU)
set(USE_CUDA ON)
set(USE_CUDNN ON)

# LLVM 활성화 (CPU 코드 생성)
set(USE_LLVM /usr/bin/llvm-config-10)

# Graph Executor 및 프로파일러
set(USE_GRAPH_EXECUTOR ON)
set(USE_PROFILER ON)

# 추가 기능
set(USE_MICRO OFF)
set(USE_RPC ON)
set(USE_SORT ON)
set(USE_RANDOM ON)

# TensorRT 연동 (선택적)
set(USE_TENSORRT_CODEGEN OFF)
set(USE_TENSORRT_RUNTIME OFF)

# 디버그 빌드 (필요시 ON)
set(USE_RELAY_DEBUG OFF)
EOF

log_info "config.cmake created"

# Step 4: CMake 설정
log_info "Step 4: Running CMake..."
cmake -G Ninja ..

# Step 5: 빌드 (6코어 사용, 메모리 고려)
log_info "Step 5: Building TVM (this may take 1-2 hours)..."
ninja -j6

# Step 6: Python 패키지 설치
log_info "Step 6: Installing Python package..."
cd "$TVM_HOME/python"
pip3 install -e .

# Step 7: 환경 변수 설정
log_info "Step 7: Setting environment variables..."
BASHRC="$HOME/.bashrc"

if ! grep -q "TVM_HOME" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# TVM Configuration" >> "$BASHRC"
    echo "export TVM_HOME=$TVM_HOME" >> "$BASHRC"
    echo "export PYTHONPATH=\$TVM_HOME/python:\$PYTHONPATH" >> "$BASHRC"
    log_info "Added TVM to .bashrc"
fi

# Step 8: 설치 확인
log_info "Step 8: Verifying installation..."
source "$BASHRC"

python3 << 'PYEOF'
import tvm
from tvm import relay
import numpy as np

print("=" * 50)
print("TVM Installation Verification")
print("=" * 50)
print(f"TVM Version: {tvm.__version__}")
print(f"CUDA Available: {tvm.cuda().exist}")

if tvm.cuda().exist:
    # 간단한 GPU 테스트
    x = tvm.nd.array(np.random.randn(3, 4).astype("float32"), tvm.cuda())
    print(f"GPU Array Test: PASSED")
    print(f"  Shape: {x.shape}")
    print(f"  Device: {x.device}")
else:
    print("WARNING: CUDA not available!")

print("=" * 50)
print("Installation completed successfully!")
print("=" * 50)
PYEOF

log_info "TVM installation complete!"
echo ""
echo "Next steps:"
echo "  1. source ~/.bashrc"
echo "  2. python3 experiments/00_tvm_setup/01_verify_install.py"
