import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from methods import autoencoder
from models.model import SimpleModel
from utils.load_dataset import get_dataloaders
from utils.plots import plot_embeddings, plot_generated_imgs, plot_reconstructed_imgs


def train(args):

    model_path = os.path.join(args.checkpoint_dir, args.name)
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    train_ds, _ = get_dataloaders(args)

    if args.method == 'autoencoder':
        model = SimpleModel(args.emb_dim)
        model = model.cuda()
        model = autoencoder.train(args, train_ds, model)

    

    model.save_model(model_path)
    print("Model Saved")

def test(args):
    
    model_path = os.path.join(args.checkpoint_dir, args.name)
    model = SimpleModel(args.emb_dim).cuda()
    model.load_model(model_path)

    _, test_ds = get_dataloaders(args)
    if args.method == 'autoencoder':
        embeddings, generated_images, labels, feat_shape = autoencoder.predict(args, test_ds, model)
    plot_embeddings(embeddings, labels)
    plot_generated_imgs(embeddings, labels, model, feat_shape)
    plot_reconstructed_imgs(test_ds, generated_images)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="FashionMNIST", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="Training dataset")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning Rate")
    parser.add_argument("--emb_dim", default=16, type=int, help="Dimension of the embeddings")
    parser.add_argument("--checkpoint_dir", default="checkpoint", help="Path to save the model weights")
    parser.add_argument("--name", default='expt', help="Name of the experiment")
    parser.add_argument("--method", default='autoencoder', choices=['autoencoder', 'vae', 'gan'], help="Method for training")
    parser.add_argument("--test", action="store_true", help="Set to true if you want to test")

    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)