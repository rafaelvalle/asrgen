# Attacking Speaker Recognition Systems with Deep Generative Models

PyTorch implementation of [Attacking Speaker Recognition Systems with Deep Generative Models](https://arxiv.org/pdf/1801.02384.pdf). 

![Real and Fake Spectrograms](demo_spectrograms.png)

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Clone this repo: `git clone https://github.com/NVIDIA/tacotron2.git`
2. CD into this repo: `cd asrgen`
3. Download the data and unzip it in this folder
4. Install python requirements: `pip install -r requirements.txt`

## Training
1. `python gan_train.py`
2. (OPTIONAL) `tensorboard --logdir=./`

## Synthesize audio samples with a Generator
1. `jupyter notebook --ip=127.0.0.1 --port=31337`
2. load `gan_synthesis.ipynb`	

## Acknowledgements
This implementation uses code from the following repos: [NVIDIA's Tacotron 2] (https://github.com/nvidia/tacotron2), [Martin Arjovsky](https://github.com/martinarjovsky/WassersteinGAN) and [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft).

We are thankful to Prem Seetharaman and Markus Rabe for their feedback on the early draft of this paper.


