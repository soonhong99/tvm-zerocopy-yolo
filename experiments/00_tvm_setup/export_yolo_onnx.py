#!/usr/bin/env python3
"""
Simplified YOLO ONNX export using PyTorch Hub
"""
import os
import torch

def export_yolo_onnx():
    """Export YOLOv5s to ONNX using PyTorch Hub"""
    print("Loading YOLOv5s from PyTorch Hub...")

    # Load model on CPU to avoid device mismatch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu')
    model.cpu()
    model.eval()

    # Prepare for export (CPU tensors)
    dummy_input = torch.randn(1, 3, 640, 640)

    # Export
    output_path = "models/yolov5s.onnx"
    os.makedirs("models", exist_ok=True)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=None
    )

    print(f"✓ Export complete: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    return output_path

if __name__ == "__main__":
    export_yolo_onnx()
