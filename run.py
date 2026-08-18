import os
import sys
import numpy as np
import torch
from model import LightDnCNN

def load_input_array(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Unsupported input shape for {os.path.basename(path)}: {arr.shape}. Expected (H, W) or (H, W, 1).")

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightDnCNN().to(device)

    weights_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "corrected_restoration_model.pth"
    )
    if not os.path.isfile(weights_path):
        print(f"Model weights not found: {weights_path}")
        sys.exit(1)

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".npy"))
    print("Device:", device)
    print("Input files:", len(files))

    if not files:
        print("No .npy files found in the input directory.")
        return

    with torch.no_grad():
        for i, filename in enumerate(files):
            noisy = load_input_array(os.path.join(input_dir, filename))
            tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
            restored = model(tensor).squeeze().cpu().numpy().astype(np.float32)
            restored = np.clip(restored, 0.0, 1.0)

            if not np.isfinite(restored).all():
                raise ValueError(f"NaN or Inf detected in output: {filename}")
            if restored.ndim != 2:
                raise ValueError(f"Output is not grayscale 2D for {filename}: {restored.shape}")

            np.save(os.path.join(output_dir, filename), restored)

            if (i + 1) % 50 == 0 or i + 1 == len(files):
                print(f"Processed: {i + 1}/{len(files)}")

    print("Restoration completed successfully.")

if __name__ == "__main__":
    main()
