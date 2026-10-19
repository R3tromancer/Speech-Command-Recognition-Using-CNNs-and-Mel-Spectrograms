# Speech Command Recognition Using CNNs

## Overview

A PyTorch-based speech recognition project that classifies **10 spoken commands** using a Convolutional Neural Network (CNN). Audio recordings are converted into Mel-spectrograms, allowing the model to learn patterns in speech and distinguish between commands.

## Features

- **Audio preprocessing:** Converts audio into fixed-length, 16 kHz signals and 64-bin Mel-spectrograms using Librosa.
- **CNN architecture:** Extracts speech features from Mel-spectrograms for command classification.
- **Efficient training:** Precomputes spectrograms and uses GPU acceleration when available.
- **Model selection:** Saves the model with the best validation accuracy.
- **Evaluation:** Measures accuracy, precision, recall, and F1-score, with a confusion matrix for class-level analysis.
- **Inference:** Predicts the command in an individual audio recording.

## Supported Commands

`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`

## Results

| Metric | Test Result |
|---|---:|
| Test accuracy | **90.28%** |
| Test samples | 4,074 |
| Macro F1-score | 90.27% |
| Weighted F1-score | 90.29% |

The model achieved its best validation accuracy of **90.85%** during 20 training epochs.

## Technology Stack

- Python
- PyTorch
- Librosa
- NumPy
- Matplotlib
- scikit-learn

## Usage

The notebook covers the complete workflow: loading the dataset, visualizing audio waveforms and Mel-spectrograms, preparing data splits, training the CNN, evaluating the best checkpoint, and predicting commands from new audio files.

## Future Improvements

- Apply audio augmentation to improve robustness to background noise and variations in pronunciation.
- Explore lightweight architectures for real-time inference.
- Extend recognition to additional words and more challenging recording conditions.
