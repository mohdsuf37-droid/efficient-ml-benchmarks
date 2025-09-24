import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import datasets, transforms

import onnxruntime as ort

from model_cnn import TinyMNISTCNN
from utils import measure_latency, memory_mb, file_size_mb

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

def get_sample_batch(n=64):
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=str(RESULTS), train=False, transform=transform, download=True)
    x = torch.stack([test_ds[i][0] for i in range(n)], dim=0)  # [n,1,28,28]
    return x

def bench_pytorch(device="cpu", batch=1):
    model = TinyMNISTCNN().to(device)
    state = RESULTS / "mnist_cnn.pt"
    assert state.exists(), "Run train.py first."
    model.load_state_dict(torch.load(state, map_location=device))
    model.eval()

    x = get_sample_batch(n=batch).to(device)

    def run():
        with torch.no_grad():
            _ = model(x)

    lat = measure_latency(run, warmup=10, runs=100)
    return {
        "backend": f"pytorch-{device}",
        "model_size_mb": file_size_mb(state),
        "mem_mb": memory_mb(),
        **lat
    }

def bench_onnx(device="cpu", batch=1):
    onnx_path = RESULTS / "mnist_cnn.onnx"
    assert onnx_path.exists(), "Run export.py first."
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        # only if CUDA build of onnxruntime is installed
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    x = get_sample_batch(n=batch).numpy()  # NHWC? we have NCHW; ONNX exported as NCHW
    # onnx expects float32 NCHW, shape [B,1,28,28]
    def run():
        _ = sess.run(["logits"], {"input": x})

    lat = measure_latency(run, warmup=10, runs=100)
    return {
        "backend": f"onnx-{device}",
        "model_size_mb": file_size_mb(onnx_path),
        "mem_mb": memory_mb(),
        **lat
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--gpu", action="store_true", help="Benchmark GPU if available (PyTorch & ONNX)")
    args = ap.parse_args()

    rows = []
    # PyTorch CPU
    rows.append(bench_pytorch(device="cpu", batch=args.batch))
    # ONNX CPU
    rows.append(bench_onnx(device="cpu", batch=args.batch))

    # Optional GPU
    if args.gpu and torch.cuda.is_available():
        rows.append(bench_pytorch(device="cuda", batch=args.batch))
        try:
            # Will work only if you installed onnxruntime-gpu
            rows.append(bench_onnx(device="cuda", batch=args.batch))
        except Exception as e:
            print(f"[WARN] ONNX GPU bench skipped: {e}")

    df = pd.DataFrame(rows)
    out_csv = RESULTS / f"bench_batch{args.batch}.csv"
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"[OK] Saved {out_csv}")
