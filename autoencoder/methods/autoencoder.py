import torch.nn as nn
from torch.optim import Adam

def train(args, train_ds, model):

    optimizer = Adam(model.parameters(), lr=args.lr)
    loss = nn.MSELoss()

    for epoch in range(args.num_epochs):
        for iter, (img, labels) in enumerate(train_ds):

            b_img = img.cuda()
            b_labels = labels.cuda()
            optimizer.zero_grad()
            emb, generated_img, _ = model(b_img)
            loss_value = loss(generated_img, b_img)
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
        emb, generated_img, feat_shape = model(b_img)
        embeddings.extend(emb.detach().cpu().numpy())
        img_list.extend(generated_img.detach().cpu().numpy())
        label_list.extend(labels)

    return embeddings, img_list, label_list, feat_shape

