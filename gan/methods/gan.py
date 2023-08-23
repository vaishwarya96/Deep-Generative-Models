import torch.nn as nn
from torch.optim import Adam
import torch

def train(args, train_ds, netG, netD):

    optimizerG = Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    real_label = 1.
    fake_label = 0.

    criterion = nn.BCELoss()

    for epoch in range(args.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_ds, 0):

            netD.zero_grad()
            # Format batch
            real_img = data[0].cuda()
            b_size = real_img.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float).cuda()
            # Forward pass real batch through D
            output = netD(real_img).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.emb_dim, 1, 1).cuda()
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, args.num_epochs, i, len(train_ds),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    print("Training finished")

    return netG, netD