import torch.nn as nn
import torch.nn.functional as F
import torch

class VAENet():
    def __init__(self, emb_dim, n_in_feat=1, filter_config=(32,64,128)):

        self.encoder = nn.ModuleList()
        self.latent = nn.ModuleList()
        self.decoder = nn.ModuleList()

        encoder_filter_config = (n_in_feat,) + filter_config
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(len(filter_config)):
            self.encoder.append(Encoder(encoder_filter_config[i],
                encoder_filter_config[i+1]))

            self.decoder.append(Decoder(decoder_filter_config[i], 
                decoder_filter_config[i+1]))

        self.decoder.append(nn.ConvTranspose2d(filter_config[0], n_in_feat, 3, 1, 1))

        self.latent.append(LatentSpace(emb_dim))

    def get_model(self):
        return self.encoder, self.decoder, self.latent

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class Encoder(nn.Module):
    def __init__(self, in_feat, out_feat):

        super(Encoder, self).__init__()

        layers = [nn.Conv2d(in_feat, out_feat, 3, 2, 1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):

        output = self.features(x)
        return output

class LatentSpace(nn.Module):
    def __init__(self, emb_dim):

        super(LatentSpace, self).__init__()

        self.flattened_layer = nn.Flatten()
        self.mean_layer = nn.Linear(2048, emb_dim)
        self.log_var_layer = nn.Linear(2048, emb_dim)
        self.sampling = Sampling()
        self.dense_layer = nn.Linear(emb_dim, 2048)

    def forward(self, x):
        feat_shape = x.shape
        x = self.flattened_layer(x)
        z_mean = self.mean_layer(x)
        z_log_var = self.log_var_layer(x)
        latent_samples = self.sampling(z_mean, z_log_var)
        x = self.dense_layer(latent_samples)
        x = x.view(feat_shape)
        return emb, x

class Decoder(nn.Module):
    def __init__(self, in_feat, out_feat):

        super(Decoder, self).__init__()

        #self.feat_shape = feat_shape
        #self.dense_layer = nn.Linear(emb_dim, 2048)
        layers = [nn.ConvTranspose2d(in_feat, out_feat, 3, 2, 1, output_padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        #output = self.dense_layer(x)
        #output = output.view(self.feat_shape)
        output = self.features(x)
        return output
