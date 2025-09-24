from PIL import Image, ImageOps
import numpy as np
import torch

def pil_to_mnist_tensor(pil_img: Image.Image):
    # convert to grayscale, resize to 28x28, white background, black digit
    pil_img = pil_img.convert("L")
    pil_img = ImageOps.invert(pil_img)  # make dark digit on light bg if needed
    pil_img = pil_img.resize((28,28))
    arr = np.array(pil_img).astype("float32") / 255.0
    arr = arr.reshape(1, 1, 28, 28)  # [B,C,H,W]
    return torch.from_numpy(arr)
