import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision.utils as vutils


from methods import gan
from utils.load_dataset import get_dataloaders
from model.model import Generator, Discriminator, weights_init

def train(args):

    model_path = os.path.join(args.checkpoint_dir, args.name)
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    #Get the train dataset
    train_ds, num_channels = get_dataloaders(args)

    # Plot some training images
    real_batch = next(iter(train_ds))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].cuda()[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


    #Define the generator and discriminator model
    netG = Generator(args.emb_dim, args.ngf, num_channels).cuda()
    netG.apply(weights_init)

    netD = Discriminator(args.ndf, num_channels).cuda()
    netD.apply(weights_init)

    #Train the model
    netG, netD = gan.train(args, train_ds, netG, netD)

    #Save the model
    generator_path = os.path.join(model_path, "generator_latest.pth")
    torch.save(netG.state_dict(), generator_path)
    discriminator_path = os.path.join(model_path, "discriminator_latest.pth")
    torch.save(netD.state_dict(), discriminator_path)


def test(args):

    #Load the model 
    model_path = os.path.join(args.checkpoint_dir, args.name)
    generator_path = os.path.join(model_path, "generator_latest.pth")

    #Load the generator and discriminator model
    netG = Generator(args.emb_dim, args.ngf, args.nc).cuda()
    netG.eval()
    netG.load_state_dict(torch.load(generator_path))

    fixed_noise = torch.randn(64, args.emb_dim, 1, 1).cuda()
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="Celeba", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10","Celeba"], help="Training dataset")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=25, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=0.0002, type=float, help="Learning Rate")
    parser.add_argument("--nc", default=3, type=int, help="Number of channels")
    parser.add_argument("--emb_dim", default=100, type=int, help="Dimension of the embeddings")
    parser.add_argument("--ngf", default=64, type=int, help="Number of feature maps in generator")
    parser.add_argument("--ndf", default=64, type=int, help="Number of feature maps in discriminator")
    parser.add_argument("--beta1", default=0.5, type=float, help="Parameter for Adam optimizer")
    parser.add_argument("--checkpoint_dir", default="checkpoint", help="Path to save the model weights")
    parser.add_argument("--name", default='expt', help="Name of the experiment")
    parser.add_argument("--test", action="store_true", help="Set to true if you want to test")

    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)
