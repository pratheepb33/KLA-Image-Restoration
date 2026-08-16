# KLA Image Restoration

Deep-learning based grayscale image restoration system for
recovering 256x256 images from 128x128 NoisyLR inputs.

## Pipeline

128x128 NoisyLR -> LightDnCNN -> 256x256 Restored Image

## Model

LightDnCNN:

- 8 convolutional denoising layers
- 32 feature channels
- ReLU activations
- PixelShuffle x2 upsampling
- Final 1-channel reconstruction layer
- Trainable parameters: 102,337

## Training

- Training pairs: 3200
- Input size: 128x128
- Target size: 256x256
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

Higher PSNR and SSIM are better.
Lower LPIPS is better.

## Runtime

End-to-end restoration of 400 test images:

- Total time: 1.228 seconds
- Average: 3.07 ms/image
- Throughput: 325.8 images/second

Runtime was measured in the CUDA/Colab environment used
for the experiment.

## Inference

Install dependencies:

    pip install -r requirements.txt

Run:

    python inference.py --input_dir /path/to/NoisyLR --output_dir /path/to/output

Optional checkpoint:

    python inference.py --input_dir /path/to/NoisyLR --output_dir /path/to/output --weights weights/corrected_restoration_model.pth

## Input

- NumPy .npy files
- Grayscale
- Shape: 128x128

## Output

- NumPy .npy files
- Grayscale
- Shape: 256x256
- Values clipped to [0, 1]

## Results

The results directory contains representative best and worst
validation examples comparing NoisyLR, Bicubic, our model,
and Ground Truth.

## Project Structure

KLA_Image_Restoration/

    model.py
    inference.py
    requirements.txt
    README.md

    configs/
        model_config.txt

    weights/
        corrected_restoration_model.pth

    results/
        best_validation_comparison.png
        worst_validation_comparison.png

    outputs/
        final_restored_outputs.zip
