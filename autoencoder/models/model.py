import torch.nn as nn
from .simple_model import SimpleNet
from .base_model import BaseModel
import torch

class SimpleModel(BaseModel):
    
    def __init__(self, emb_dim):
        super(SimpleModel, self).__init__()

        model = SimpleNet(emb_dim)
        self.encoder, self.decoder, self.latent = model.get_model()

    def forward(self, x):
        feat = x
        
        for i in range(len(self.encoder)):
            feat = self.encoder[i](feat)
        
        feat_shape = feat.shape
        emb, feat = self.latent[0](feat)

        for i in range(len(self.decoder)):
            feat = self.decoder[i](feat)

        return emb, feat, feat_shape
