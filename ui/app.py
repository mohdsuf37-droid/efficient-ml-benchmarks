import io
import sys
from pathlib import Path
# Add project root (the folder that has src/ and ui/) to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

import torch
from src.model_cnn import TinyMNISTCNN
from src.preprocess_mnist import pil_to_mnist_tensor

import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

st.set_page_config(page_title="Efficient ML Benchmarks", page_icon="🔋", layout="wide")
st.title("🔋 Efficient ML Inference Benchmarks")
st.caption("PyTorch vs ONNX Runtime on MNIST (optionally GPU). Upload a digit to test prediction & see latency.")

# Sidebar
st.sidebar.header("Options")
backend = st.sidebar.selectbox("Backend", ["pytorch-cpu", "onnx-cpu"])  # add GPU later if you want
batch = st.sidebar.number_input("Batch size", 1, 64, value=1)
run_btn = st.sidebar.button("Run Benchmark (single inference)")

# Load models
pt_state = RESULTS / "mnist_cnn.pt"
onnx_path = RESULTS / "mnist_cnn.onnx"

pt_model = None
ort_sess = None

if backend.startswith("pytorch"):
    if not pt_state.exists():
        st.error("Missing model weights. Run training first: `python src/train.py`")
    else:
        pt_model = TinyMNISTCNN()
        pt_model.load_state_dict(torch.load(pt_state, map_location="cpu"))
        pt_model.eval()

if backend.startswith("onnx"):
    if not onnx_path.exists():
        st.error("Missing ONNX model. Export first: `python src/export.py`")
    else:
        ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

# Upload image
st.subheader("Upload a digit (28x28-ish, white background works best)")
file = st.file_uploader("PNG/JPG", type=["png","jpg","jpeg"])
if file is not None:
    pil = Image.open(file)
    st.image(pil, caption="Uploaded", width=120)
else:
    st.info("Tip: You can also draw a digit in MS Paint/Photos, save, and upload.")

# Predict
if run_btn:
    if file is None:
        st.warning("Please upload an image first.")
    else:
        x = pil_to_mnist_tensor(pil)  # [1,1,28,28]
        x = x.repeat(batch,1,1,1)

        if backend == "pytorch-cpu" and pt_model is not None:
            with torch.no_grad():
                # simple timing
                import time
                t0 = time.perf_counter()
                logits = pt_model(x)
                t1 = time.perf_counter()
                pred = logits.argmax(dim=1).cpu().numpy()
                ms = (t1 - t0) * 1000
                st.success(f"PyTorch CPU: batch={batch}, time={ms:.2f} ms, preds (first 8): {pred[:8]}")
        elif backend == "onnx-cpu" and ort_sess is not None:
            import time
            t0 = time.perf_counter()
            logits = ort_sess.run(["logits"], {"input": x.numpy()})[0]
            t1 = time.perf_counter()
            pred = np.argmax(logits, axis=1)
            ms = (t1 - t0) * 1000
            st.success(f"ONNX CPU: batch={batch}, time={ms:.2f} ms, preds (first 8): {pred[:8]}")
        else:
            st.error("Selected backend is not ready. Train/export first.")

# Show last benchmark CSVs if exist
st.subheader("Latest Benchmarks")
csvs = sorted(RESULTS.glob("bench_batch*.csv"))
if csvs:
    latest = csvs[-1]
    st.write(f"Showing: `{latest.name}`")
    df = pd.read_csv(latest)
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("backend")["mean_ms"])
else:
    st.info("Run a benchmark to generate results: `python src/bench.py --batch 1`")

st.caption("Tip: Install `onnxruntime-gpu` and add GPU runs in `src/bench.py` to compare GPU vs CPU.")
