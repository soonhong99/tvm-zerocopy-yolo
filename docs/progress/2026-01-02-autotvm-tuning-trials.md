# TVM Auto-tuning Trials and Progress Report

**Date:** 2026-01-02
**Platform:** NVIDIA Jetson Xavier NX / AGX Xavier
**Model:** YOLOv5s (ONNX)
**Objective:** Optimize YOLOv5s inference latency using TVM Auto-tuning (AutoTVM).

---

## 1. Summary of Attempts

We aimed to optimize the YOLOv5s model to achieve performance comparable to or better than TensorRT (~30ms) using TVM's Auto-tuning capabilities. We explored both Cross-compilation (RPC) and On-device tuning methods.

| Attempt | Method | Environment | Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Attempt 1** | Cross-compilation (RPC) | Windows (WSL) | Failed | ❌ Issues with WSL environment & RPC connection |
| **Attempt 2** | Cross-compilation (RPC) | macOS (Apple M1) | Failed | ❌ Host lacks CUDA support (cannot build CUDA kernels) |
| **Attempt 3** | Docker Environment | Jetson (Docker) | Failed | ❌ Auto-tuning instability in containerized env |
| **Attempt 4** | On-device Tuning (Ultra) | Jetson (Native) | **Regression** (112ms → 145ms) | ⚠️ Completed but performance degraded |
| **Attempt 5** | On-device Tuning (Full) | Jetson (Native) | **In Progress** | ⏳ Running now (Expected: ~30-40ms) |

---

## 2. Detailed Analysis of Failures

### 2.1. Cross-compilation via RPC (Windows/macOS)
To accelerate the tuning process (which is CPU-intensive), we attempted to use a powerful host machine (PC/Mac) to generate tuning configurations and send them to the Jetson device via RPC.

*   **Windows (WSL):** Encountered significant friction with WSL networking and environment isolation, preventing stable RPC communication with the Jetson board.
*   **macOS (Apple M1):**
    *   **Issue:** AutoTVM requires the **host machine** to be able to compile the target code.
    *   **Blocker:** Since the target is NVIDIA GPU (`cuda`), the host must have CUDA Toolkit installed. macOS M1 does not support CUDA, making it impossible to generate CUDA kernels for the Jetson.
    *   **Conclusion:** Cross-compilation from non-NVIDIA hosts to NVIDIA targets is limited for AutoTVM.

### 2.2. On-device Tuning: Ultra Mode (Quick Test)
We switched to running the tuning directly on the Jetson Xavier to bypass host limitations. We used the "Ultra" mode for a quick assessment.

*   **Configuration:**
    *   **Scope:** Top 10 most computationally expensive layers (by FLOPs).
    *   **Tuner:** `RandomTuner` (Fast but less efficient).
    *   **Trials:** 100 trials per task.
*   **Results:**
    *   **Baseline (TVM Default):** 112.56 ms
    *   **Tuned (Ultra Mode):** 144.91 ms 📉
*   **Root Cause Analysis:**
    1.  **Layout Mismatch Overhead:** Tuning only a subset (Top 10) of layers forces TVM to insert **Layout Transform** operations between tuned layers (optimized layout) and untuned layers (default layout). This data reordering cost outweighed the computation gains.
    2.  **Random Tuner Limitations:** `RandomTuner` with few trials likely selected suboptimal configurations that were worse than the highly optimized default CuDNN/Cublas kernels.

---

## 3. Current Strategy: Full On-device Tuning

To resolve the performance regression and achieve maximum speedup, we are currently executing a **Full Tuning** session on the Jetson Xavier.

*   **Configuration:**
    *   **Scope:** **All layers** (eliminates layout mismatch).
    *   **Tuner:** `XGBTuner` (Machine Learning-based, learns from previous trials).
    *   **Trials:** 1500 trials per task.
*   **Expected Outcome:**
    *   Elimination of layout transform overheads.
    *   Discovery of optimal kernel configurations surpassing default libraries.
    *   Target Latency: **30ms - 40ms** (Competitive with TensorRT).

---

## 4. Next Steps
1.  Wait for Full Tuning completion (estimated: several hours).
2.  Benchmark the fully tuned model against TensorRT.
3.  Integrate the optimized model into the End-to-End pipeline (`04_e2e_pipeline.py`).
