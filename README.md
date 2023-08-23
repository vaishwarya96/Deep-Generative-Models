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

## Variational Autoencoder
```bash
cd vae/
```

For training, run the following command:
```bash
python3 main.py
```
For inference, run
```bash
python3 main.py --test
```

## Generative Adversarial Networks
```bash
cd gan/
```
To train on CelebA, download the dataset from <a href="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" target="_blank">this site</a>. The dataset will download as a file named `img_align_celeba.zip`. Once downloaded, create a directory named `data/celeba` and extract the zip file into that directory.

The structure is `data/celeba/img_align_celeba/(Images)`

For training, run the command:
```bash
python3 main.py
```
For inference, run
```bash
python3 main.py --test
```


