# Deep-Generative-Models

This repository contains demo code in PyTorch for four deep generative models:
1. Autoencoders
2. Variational Autoencoders (VAEs)
3. Generative Adversarial Networks (GANs)
4. Denoising Diffusion Models

To use this repository:
```bash
git clone https://github.com/vaishwarya96/Deep-Generative-Models.git
```
```bash
cd Deep-Generative-Models/
```

## Autoencoder

```bash
cd autoencoder/
```

For training an autoencoder, run the following command:

```bash
python3 main.py
```
For inference, run the following command:
```bash
python3 main.py --test
```
Note: By default, the network trains on the FashionMNIST dataset. If you want to train on other datasets, you can do so by changing the argument. For example, to train on MNIST, run the following command:
```bash
python3 main.py --dataset MNIST
```
Similarly, you can adjust the hyperparameters by passing the values in the arguments. Check the file `main.py` for the list of arguments and their default values.
