# KLA Image Restoration

AI-based restoration of degraded grayscale images for semiconductor inspection.

This solution restores 128x128 NoisyLR grayscale `.npy` inputs to 256x256 restored grayscale `.npy` outputs using a lightweight LightDnCNN-based deep learning model.

## Model

LightDnCNN:
- 8 convolutional denoising layers
- 32 feature channels
- ReLU activations
- PixelShuffle x2 upsampling
- Final 1-channel reconstruction layer
- Trainable parameters: 102,337

## Dataset and Training

- Training pairs: 3200
- Input resolution: 128x128
- Target resolution: 256x256
- Batch size: 16
- Epochs: 5
- Loss: L1Loss
- Optimizer: Adam
- Learning rate: 1e-4

## Validation Results

Validation set: 100 paired samples.

| Metric | Our Model | Bicubic |
|---|---:|---:|
| PSNR | 27.63 dB | 23.79 dB |
| SSIM | 0.746 | 0.575 |
| LPIPS | 0.361 | 0.398 |

Higher PSNR and SSIM are better. Lower LPIPS is better.

## Runtime

End-to-end restoration of 400 test images:

- Total time: 1.228 seconds
- Average: 3.07 ms/image
- Throughput: 325.8 images/second

Runtime was measured in the CUDA environment used for the experiment.

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
