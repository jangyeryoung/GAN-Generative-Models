# Text Classification using TextCNN
This repository contains a collection of Generative Adversarial Networks (GANs) implemented in PyTorch. 
It demonstrates the progression of generative models from the basic Vanilla GAN to more advanced architectures like DCGAN and CGAN.

## Overview
This project was developed as part of a university Deep Learning course.

I have implemented a comprehensive pipeline that allows for training and evaluating various GAN architectures.

## Features
- Model Implementations:
	- Vanilla GAN: Basic GAN structure using Fully Connected layers.
	- DCGAN: Improved stability and quality using Convolutional layers and architectural symmetry.
	- CGAN: Conditional image generation by integrating class labels via feature concatenation.
- Pipeline:
	- Modular training loop capable of switching between models.
	- Integrated loss functions (G_Loss, D_Loss) for stable training.

## Note
The core architecture of this project is based on course materials provided by the university.
I have independently implemented the data preprocessing pipeline, conducted hyperparameter tuning, and developed the training log monitoring system.

Designed with architectural symmetry between the Generator and Discriminator, using Transposed Convolutions for stable upsampling. Applied weight initialization based on the DCGAN paper (Normal distribution with std=0.02) to ensure rapid convergence and prevent vanishing gradients.

Implemented a suite of Generative Adversarial Networks (GANs), ranging from Vanilla GAN to DCGAN and CGAN. 
Expertly handled conditional generation by integrating class labels via feature concatenation, enabling targeted image synthesis.
