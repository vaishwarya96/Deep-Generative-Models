import torch.nn as nn
from torch.optim import Adam
import torch

def train(args, train_ds, model):

    optimizer = Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    for epoch in range(args.num_epochs):
        for iter, (img, labels) in enumerate(train_ds):

            b_img = img.cuda()
            b_labels = labels.cuda()
            optimizer.zero_grad()
            z_mean, z_log_var, z, generated_img, _ = model(b_img)
            reconstruction_loss = mse_loss(generated_img, b_img)
            kl_loss = torch.mean(torch.sum(-0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp()), dim=1))
            loss_value = 50*reconstruction_loss + kl_loss
            loss_value.backward()
            optimizer.step()
        print("Epoch: %d, loss: %f" %(epoch+1, loss_value.item()))

    print("Training finished")

    return model

def predict(args, test_ds, model):

    img_list = []
    embeddings = []
    label_list = []
    for iter, (img,labels) in enumerate(test_ds):
        b_img = img.cuda()
        b_labels = labels.cuda()
        z_mean, z_log_var, z, generated_img, feat_shape = model(b_img)
        embeddings.extend(z.detach().cpu().numpy())
        img_list.extend(generated_img.detach().cpu().numpy())
        label_list.extend(labels)

    return embeddings, img_list, label_list, feat_shape

