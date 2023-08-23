import torch.nn as nn
from .vae_model import VAENet
from .base_model import BaseModel

class VAEModel(BaseModel):
    
    def __init__(self, emb_dim):
        super(VAEModel, self).__init__()

        model = VAENet(emb_dim)
        self.encoder, self.decoder, self.latent = model.get_model()

    def forward(self, x):
        feat = x
        
        for i in range(len(self.encoder)):
            feat = self.encoder[i](feat)
        
        feat_shape = feat.shape
        z_mean, z_log_var, z, feat = self.latent[0](feat)

        for i in range(len(self.decoder)):
            feat = self.decoder[i](feat)

        return z_mean, z_log_var, z, feat, feat_shape