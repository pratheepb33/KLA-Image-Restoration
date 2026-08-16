
import argparse
import os
import numpy as np
import torch

from model import LightDnCNN


def main():

    parser = argparse.ArgumentParser(
        description="KLA 128x128 to 256x256 image restoration"
    )

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--weights",
        default="weights/corrected_restoration_model.pth"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = LightDnCNN().to(device)

    checkpoint = torch.load(
        args.weights,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith(".npy")
    ])

    print("Device:", device)
    print("Input files:", len(files))

    for i, filename in enumerate(files):

        input_path = os.path.join(args.input_dir, filename)

        noisy = np.load(input_path).astype(np.float32)

        tensor = (
            torch.from_numpy(noisy)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            restored = model(tensor)

        restored = restored.squeeze().cpu().numpy()
        restored = np.clip(restored, 0.0, 1.0)

        output_path = os.path.join(
            args.output_dir,
            filename
        )

        np.save(
            output_path,
            restored.astype(np.float32)
        )

        if (i + 1) % 50 == 0:
            print(f"Processed: {i + 1}/{len(files)}")

    print("Restoration completed.")


if __name__ == "__main__":
    main()
