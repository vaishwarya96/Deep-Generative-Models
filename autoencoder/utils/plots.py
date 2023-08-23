import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_embeddings(embeddings, labels):

    embeddings = np.array(embeddings)
    figsize = 8
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        cmap="rainbow",
        c=labels,
        alpha=0.8,
        s=3,
    )
    plt.colorbar()
    plt.title("Latent Space")
    plt.show()

def plot_generated_imgs(embeddings, labels, model, feat_shape):

    embeddings = np.array(embeddings)
    emb_dim = embeddings[0].shape[0]

    mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)

    grid_width, grid_height = (6, 3)
    sample = torch.Tensor(np.random.uniform(
        mins, maxs, size=(grid_width * grid_height, emb_dim)
    )).cuda()
    
    dense = model.latent[0].dense_layer(sample)
    feat = dense.view(-1, feat_shape[1], feat_shape[2], feat_shape[3])
    
    for i in range(len(model.decoder)):
        feat = model.decoder[i](feat)

    images = feat.detach().cpu().numpy()
    images = np.transpose(images, (0,2,3,1))
    
    figsize = 8
    plt.figure(figsize=(figsize, figsize))

    # ... the original embeddings ...
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=2)

    sample = sample.cpu().numpy()
    # ... and the newly generated points in the latent space
    plt.scatter(sample[:, 0], sample[:, 1], c="#00B0F0", alpha=1, s=40)
    plt.show()

    fig = plt.figure(figsize=(figsize, grid_height * 2))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(grid_width * grid_height):
        ax = fig.add_subplot(grid_height, grid_width, i + 1)
        ax.axis("off")
        #ax.text(
        #    0.5,
        #    -0.35,
        #    str(np.round(sample[i, :], 1)),
        #    fontsize=10,
        #    ha="center",
        #    transform=ax.transAxes,
        #)
        ax.imshow(images[i, :, :], cmap="Greys")
    plt.show()

    '''
    # Colour the embeddings by their label (clothing type - see table)
    figsize = 12
    grid_size = 15
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        cmap="rainbow",
        c=labels,
        alpha=0.8,
        s=3,
    )
    plt.colorbar()

    x = np.linspace(min(embeddings[:, 0]), max(embeddings[:, 0]), grid_size)
    y = np.linspace(max(embeddings[:, 1]), min(embeddings[:, 1]), grid_size)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    grid = torch.Tensor(np.array(list(zip(xv, yv)))).cuda()

    dense = model.latent[0].dense_layer(grid)
    feat = dense.view(-1, feat_shape[1], feat_shape[2], feat_shape[3])
    
    for i in range(len(model.decoder)):
        feat = model.decoder[i](feat)

    images = feat.detach().cpu().numpy()
    images = np.transpose(images, (0,2,3,1))
    grid = grid.cpu().numpy()
    plt.scatter(grid[:, 0], grid[:, 1], c="black", alpha=1, s=10)
    plt.show()

    fig = plt.figure(figsize=(figsize, figsize))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(grid_size**2):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.axis("off")
        ax.imshow(images[i, :, :], cmap="Greys")
    plt.show()
    '''

def plot_reconstructed_imgs(test_ds, generated_imgs):

    indices = np.random.choice(np.arange(len(generated_imgs)), 16)
    selected_generated_imgs = [generated_imgs[idx].transpose(1,2,0) for idx in indices]
    selected_gt = [test_ds.dataset[idx][0].detach().cpu().numpy().transpose(1,2,0) for idx in indices]

    figsize = 8
    grid_size = 4

    fig = plt.figure(figsize=(figsize, figsize))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(grid_size**2):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.axis("off")
        ax.imshow(selected_gt[i], cmap="Greys")
    plt.show()

    fig = plt.figure(figsize=(figsize, figsize))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(grid_size**2):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.axis("off")
        ax.imshow(selected_generated_imgs[i], cmap="Greys")
    plt.show()
