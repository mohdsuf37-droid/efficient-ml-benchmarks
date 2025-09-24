# 🔋 Efficient ML Inference Benchmarks

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)]()
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-005CED?logo=onnx&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)]()
[![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)]()
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

**Benchmark ML inference efficiency** across **PyTorch (CPU/GPU)** and **ONNX Runtime**, with metrics for **latency (p50/p90/mean)**, **memory**, and **model size** — plus a **Streamlit dashboard** where you can upload a digit and compare backends interactively.

---

## ✨ What’s inside

- Tiny CNN trained on **MNIST**
- Export to **ONNX**
- Benchmarks:
  - **PyTorch (CPU)**, **ONNX Runtime (CPU)**
  - Optional: **GPU** (if you install `onnxruntime-gpu`)
- Metrics:
  - **Latency** (mean / p50 / p90)
  - **Memory** (RSS MB)
  - **Model size** (MB)
- **Streamlit** app to upload an image and test single-inference latency + see benchmark CSVs

---

## 🚀 Quickstart

# 1) Setup
python -m venv venv
# Windows:
.\venv\Scripts\Activate
# macOS/Linux:
# source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Train on MNIST (auto-downloads)
python src/train.py

# 3) Export to ONNX
python src/export.py

# 4) Run benchmarks (CPU)
python src/bench.py --batch 1

# 5) Launch Streamlit dashboard
streamlit run ui/app.py

GPU comparison: install onnxruntime-gpu, then run:
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
python src/bench.py --batch 1 --gpu

## 🗂 Project Structure

efficient-ml-benchmarks/
-├─ README.md
-├─ requirements.txt
-├─ .gitignore
-├─ src/
-│  ├─ train.py               # train a small CNN on MNIST
-│  ├─ export.py              # export to ONNX (and save .pt)
-│  ├─ bench.py               # benchmarks (PyTorch vs ONNX Runtime)
-│  ├─ utils.py               # timing, memory, file size helpers
-│  ├─ model_cnn.py           # tiny CNN
-│  └─ preprocess_mnist.py    # preprocess uploaded images
-├─ results/
-│  └─ bench_batch1.csv (etc.)
-└─ ui/
-  └─ app.py                 # Streamlit dashboard

# 📊 Outputs

-results/mnist_cnn.pt – PyTorch weights

-results/mnist_cnn.onnx – ONNX model

-results/bench_batch.csv* – benchmark metrics
-Example columns: backend, model_size_mb, mem_mb, p50_ms, p90_ms, mean_ms

-The Streamlit app also shows a bar chart of mean latency by backend.

# 🪪 License
MIT — free to use and adapt.