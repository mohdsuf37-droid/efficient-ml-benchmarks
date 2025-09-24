import torch
from pathlib import Path
from model_cnn import TinyMNISTCNN

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

if __name__ == "__main__":
    device = "cpu"
    model = TinyMNISTCNN()
    state_path = RESULTS / "mnist_cnn.pt"
    assert state_path.exists(), "Run train.py first to create results/mnist_cnn.pt"
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    dummy = torch.randn(1,1,28,28)
    onnx_path = RESULTS / "mnist_cnn.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12
    )
    print(f"[OK] Exported ONNX to {onnx_path}")
