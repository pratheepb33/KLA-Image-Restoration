
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import LightDnCNN


# ============================
# CONFIG
# ============================

NOISY_DIR = "/content/train_extracted/train/NoisyLR"
GT_DIR = "/content/train_extracted/train/GT"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4


# ============================
# DATASET
# ============================

class NPYRestorationDataset(Dataset):

    def __init__(self, noisy_dir, gt_dir):

        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        self.files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        noisy = np.load(
            os.path.join(self.noisy_dir, filename)
        ).astype(np.float32)

        gt = np.load(
            os.path.join(self.gt_dir, filename)
        ).astype(np.float32)

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt


# ============================
# TRAIN
# ============================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataset = NPYRestorationDataset(
        NOISY_DIR,
        GT_DIR
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print("Training images:", len(dataset))
    print("Batches:", len(loader))

    model = LightDnCNN().to(device)

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for noisy, gt in loader:

            noisy = noisy.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            output = model(noisy)

            loss = criterion(output, gt)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {avg_loss:.6f}"
        )

    os.makedirs("weights", exist_ok=True)

    save_path = "weights/corrected_restoration_model.pth"

    torch.save(
        model.state_dict(),
        save_path
    )

    print("\nTraining completed.")
    print("Model saved:", save_path)


if __name__ == "__main__":
    main()
