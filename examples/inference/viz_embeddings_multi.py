#!/usr/bin/env python
# coding: utf-8

'''
Similar to viz_embeddings.py and applies the same code but to a directory of .wav files
'''

import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import src.models as models
from IPython.display import Audio, display
import csv

# Convenient for importing modules from the parent directory
import sys
sys.path.append("../../")

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Function to process and plot each audio file
def process_and_plot(audio_path, model, data_args, model_args, output_dir):
    # Load and normalize the waveform
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform - waveform.mean()

    # Extract the features
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=data_args.num_mel_bins,
        dither=0.0,
        frame_shift=10
    )

    # Compute the padding length
    n_frames = fbank.shape[0]
    p = data_args.target_length - n_frames

    # Cut or pad the fbank to the target length
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:data_args.target_length, :]

    # Normalize the features
    fbank = (fbank - data_args.mean) / (data_args.std * 2)

    # Add batch dimension
    fbank = fbank.unsqueeze(0)

    # Move to device
    fbank = fbank.to(model_args.device)

    # Forward pass to get all token embeddings
    with torch.no_grad():
        all_token_embeddings = model(fbank, return_features=True)

    # Convert embeddings to CPU and numpy
    all_token_embeddings = all_token_embeddings.cpu().numpy()

    # Convert fbank to numpy and remove the batch dimension
    fbank_np = fbank.squeeze(0).cpu().numpy()  # Shape: (128, 1024)

    # Number of patches along each dimension
    H_patches, W_patches = 8, 64  # Number of patches along height and width
    B, N, D = all_token_embeddings.shape

    # Ensure that the number of patches matches the expected shape
    assert N == H_patches * W_patches, "Mismatch in number of patches"

    # Reshape to (B, H_patches, W_patches, D)
    patch_grid = all_token_embeddings.reshape(B, H_patches, W_patches, D)

    # Reshape to (total_patches, D) for dimensionality reduction
    patch_embeddings = patch_grid.reshape(-1, D)  # Shape: (H_patches * W_patches, D)

    # Reduce dimensions to 1 using PCA
    pca = PCA(n_components=1)
    reduced_embeddings = pca.fit_transform(patch_embeddings)  # Shape: (H_patches * W_patches, 1)

    # Reshape back to grid
    reduced_grid = reduced_embeddings.reshape(H_patches, W_patches)  # Shape: (H_patches, W_patches)

    # Option 3: Overlay the PCA-Reduced Embeddings on the Spectrogram
    plt.figure(figsize=(12, 6))
    plt.imshow(np.transpose(fbank_np), aspect='auto', origin='lower', cmap='viridis', alpha=0.7)
    plt.imshow(reduced_grid, cmap='plasma', alpha=0.3, extent=(0, 1024, 0, 128), aspect='auto', origin='lower')
    plt.colorbar(label='Magnitude / PCA Reduced Value')
    plt.title('Mel-Spectrogram with Overlayed PCA-Reduced Embeddings')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')

    # Save the plot
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    plt.savefig(os.path.join(output_dir, f"{filename}_overlay_plot.png"))
    plt.close()

# Main setup
if __name__ == '__main__':
    # Define paths
    audio_dir = '/home/jdcast/wav_training_data/TK/wav/'  # Replace with your directory containing audio files
    output_dir = '/home/jdcast/wav_training_data/TK/visualizations/embeddings/AuM/mel_spectro_and_patch_overlays/'  # Replace with your desired output directory
    os.makedirs(output_dir, exist_ok=True)

    # Model and data setup
    data_args = Namespace(
        num_mel_bins=128,
        target_length=1024,
        mean=-5.0767093,
        std=4.4533687,
    )

    model_args = Namespace(
        model_type='base',
        n_classes=309,
        imagenet_pretrain=False,
        imagenet_pretrain_path=None,
        aum_pretrain=True,
        aum_pretrain_path='models/aum-base_audioset-vggsound.pth',
        aum_variant='Fo-Bi',
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # Initialize the model
    if 'base' in model_args.model_type:
        embed_dim = 768
    elif 'small' in model_args.model_type:
        embed_dim = 384
    elif 'tiny' in model_args.model_type:
        embed_dim = 192

    bimamba_type = {
        'Fo-Fo': 'none',
        'Fo-Bi': 'v1',
        'Bi-Bi': 'v2'
    }.get(model_args.aum_variant, None)

    AuM = models.AudioMamba(
        spectrogram_size=(data_args.num_mel_bins, data_args.target_length),
        patch_size=(16, 16),
        strides=(16, 16),
        embed_dim=embed_dim,
        num_classes=model_args.n_classes,
        imagenet_pretrain=model_args.imagenet_pretrain,
        imagenet_pretrain_path=model_args.imagenet_pretrain_path,
        aum_pretrain=model_args.aum_pretrain,
        aum_pretrain_path=model_args.aum_pretrain_path,
        bimamba_type=bimamba_type,
    )

    AuM.to(model_args.device)
    AuM.eval()

    # Process each audio file in the directory
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith('.wav'):  # Assuming files are in .wav format
            audio_path = os.path.join(audio_dir, audio_file)
            process_and_plot(audio_path, AuM, data_args, model_args, output_dir)