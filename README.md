# Deep Learning-Based Multiclass Game Action Recognition from Gameplay Images

Dissertation project code for classifying 10 game/sport actions from image frames, comparing a from-scratch Custom CNN against three pretrained architectures (VGG16, ResNet50, EfficientNetB0) via two-phase transfer learning.

## Repository contents

- `prepare_dataset_colab_v2.py` — downloads UCF101, extracts the 10 relevant classes, samples frames at 1fps, and produces a stratified 70:15:15 train/val/test split. Designed to run cell-by-cell in Google Colab and is resumable (safe against Colab session disconnects).
- `train_two_phase_v2.py` — builds and trains all 4 models (Custom CNN, VGG16, ResNet50, EfficientNetB0) using two-phase fine-tuning, then evaluates each on the held-out test set and saves training logs, a classification report, and a confusion matrix per model.

## Dataset

This repository does **not** include the dataset itself. The 10-class, ~10,000-image subset is derived from UCF101 (Soomro, Zamir and Shah, 2012), which is distributed by the University of Central Florida's Center for Research in Computer Vision at https://www.crcv.ucf.edu/data/UCF101.php.

Run `prepare_dataset_colab_v2.py` against the official UCF101 archive to regenerate the exact same dataset used in this study (fixed random seed = 42 for the train/val/test split).

Classes used: BenchPress, Basketball, Biking, Fencing, GolfSwing, HorseRiding, Kayaking, Skiing, SoccerJuggling, TennisSwing.

## Model weights

Trained model weights (`best_model.keras` per architecture) are not included in this repository due to file size. Available on request, or via Drive link (add link here).

## Environment

Python 3.10, TensorFlow 2.12, trained via Google Colab (GPU runtime).

## Reference

Soomro, K., Zamir, A.R. and Shah, M. (2012) *UCF101: A dataset of 101 human action classes from videos in the wild.* CRCV-TR-12-01.
