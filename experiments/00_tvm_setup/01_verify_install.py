#!/usr/bin/env python3
"""
TVM 설치 확인 및 기본 테스트
실행: python3 experiments/00_tvm_setup/01_verify_install.py
"""

import sys
import time
import numpy as np

def check_tvm_import():
    """TVM 임포트 확인"""
    print("\n[1/5] TVM Import Check")
    print("-" * 40)
    try:
        import tvm
        from tvm import relay, te
        from tvm.contrib import graph_executor
        print(f"  TVM Version: {tvm.__version__}")
        print(f"  Import: PASSED")
        return True
    except ImportError as e:
        print(f"  Import: FAILED - {e}")
        return False


def check_cuda():
    """CUDA 가용성 확인"""
    print("\n[2/5] CUDA Check")
    print("-" * 40)
    import tvm

    cuda_available = tvm.cuda().exist
    print(f"  CUDA Available: {cuda_available}")

    if cuda_available:
        # GPU 메모리 테스트
        try:
            x = tvm.nd.array(np.random.randn(1000, 1000).astype("float32"), tvm.cuda())
            print(f"  GPU Memory Test: PASSED")
            print(f"  Array Shape: {x.shape}")
            return True
        except Exception as e:
            print(f"  GPU Memory Test: FAILED - {e}")
            return False
    return False


def check_simple_compute():
    """간단한 연산 컴파일 테스트"""
    print("\n[3/5] Simple Compute Test")
    print("-" * 40)

    import tvm
    from tvm import te

    try:
        # 벡터 덧셈 정의
        n = 1024
        A = te.placeholder((n,), name="A", dtype="float32")
        B = te.placeholder((n,), name="B", dtype="float32")
        C = te.compute((n,), lambda i: A[i] + B[i], name="C")

        # 스케줄 생성
        s = te.create_schedule(C.op)

        # GPU 스레드 매핑
        bx, tx = s[C].split(C.op.axis[0], factor=64)
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))

        # 빌드
        target = "cuda -arch=sm_72"
        func = tvm.build(s, [A, B, C], target=target, name="vector_add")

        # 실행
        dev = tvm.cuda()
        a = tvm.nd.array(np.random.randn(n).astype("float32"), dev)
        b = tvm.nd.array(np.random.randn(n).astype("float32"), dev)
        c = tvm.nd.array(np.zeros(n, dtype="float32"), dev)

        func(a, b, c)

        # 검증
        expected = a.numpy() + b.numpy()
        np.testing.assert_allclose(c.numpy(), expected, rtol=1e-5)

        print(f"  Vector Add Compile: PASSED")
        print(f"  Numerical Check: PASSED")
        return True

    except Exception as e:
        print(f"  Simple Compute: FAILED - {e}")
        return False


def check_relay():
    """Relay IR 테스트"""
    print("\n[4/5] Relay IR Test")
    print("-" * 40)

    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor

    try:
        # 간단한 Relay 그래프 생성
        x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
        weight = relay.var("weight", shape=(64, 3, 7, 7), dtype="float32")
        y = relay.nn.conv2d(x, weight, strides=(2, 2), padding=(3, 3))
        y = relay.nn.relu(y)

        func = relay.Function([x, weight], y)
        mod = tvm.IRModule.from_expr(func)

        # 컴파일
        target = tvm.target.Target("cuda -arch=sm_72")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target)

        # 실행
        dev = tvm.cuda()
        module = graph_executor.GraphModule(lib["default"](dev))

        # 입력 설정
        x_data = np.random.randn(1, 3, 224, 224).astype("float32")
        weight_data = np.random.randn(64, 3, 7, 7).astype("float32")

        module.set_input("x", x_data)
        module.set_input("weight", weight_data)
        module.run()

        output = module.get_output(0).numpy()
        print(f"  Relay Compile: PASSED")
        print(f"  Output Shape: {output.shape}")
        return True

    except Exception as e:
        print(f"  Relay Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_matmul():
    """행렬 곱셈 벤치마크"""
    print("\n[5/5] Matrix Multiplication Benchmark")
    print("-" * 40)

    import tvm
    from tvm import te

    try:
        M, K, N = 1024, 1024, 1024

        # 연산 정의
        A = te.placeholder((M, K), name="A", dtype="float32")
        B = te.placeholder((K, N), name="B", dtype="float32")
        k = te.reduce_axis((0, K), name="k")
        C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

        # 스케줄 (기본)
        s = te.create_schedule(C.op)

        # 타일링
        block_size = 32
        i, j = s[C].op.axis
        k_axis = s[C].op.reduce_axis[0]

        i_outer, i_inner = s[C].split(i, factor=block_size)
        j_outer, j_inner = s[C].split(j, factor=block_size)

        s[C].reorder(i_outer, j_outer, i_inner, j_inner, k_axis)
        s[C].bind(i_outer, te.thread_axis("blockIdx.y"))
        s[C].bind(j_outer, te.thread_axis("blockIdx.x"))
        s[C].bind(i_inner, te.thread_axis("threadIdx.y"))
        s[C].bind(j_inner, te.thread_axis("threadIdx.x"))

        # 빌드
        func = tvm.build(s, [A, B, C], target="cuda -arch=sm_72", name="matmul")

        # 데이터 준비
        dev = tvm.cuda()
        a = tvm.nd.array(np.random.randn(M, K).astype("float32"), dev)
        b = tvm.nd.array(np.random.randn(K, N).astype("float32"), dev)
        c = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)

        # 워밍업
        for _ in range(5):
            func(a, b, c)
        dev.sync()

        # 벤치마크
        num_runs = 20
        start = time.time()
        for _ in range(num_runs):
            func(a, b, c)
        dev.sync()
        elapsed = time.time() - start

        avg_time = elapsed / num_runs * 1000
        gflops = (2 * M * K * N) / (avg_time / 1000) / 1e9

        print(f"  Matrix Size: {M}x{K} @ {K}x{N}")
        print(f"  Average Time: {avg_time:.2f} ms")
        print(f"  Performance: {gflops:.1f} GFLOPS")
        return True

    except Exception as e:
        print(f"  Benchmark: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 50)
    print("TVM Installation Verification")
    print("=" * 50)

    results = []
    results.append(("TVM Import", check_tvm_import()))

    if results[-1][1]:  # TVM import 성공시에만 계속
        results.append(("CUDA", check_cuda()))
        results.append(("Simple Compute", check_simple_compute()))
        results.append(("Relay IR", check_relay()))
        results.append(("Benchmark", benchmark_matmul()))

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("\nAll tests passed! TVM is ready for use.")
        print("\nNext steps:")
        print("  1. Run baseline TensorRT benchmark")
        print("  2. Start Zero-Copy experiments")
        return 0
    else:
        print("\nSome tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
