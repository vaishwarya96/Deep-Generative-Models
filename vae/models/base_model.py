import torch.nn as nn
import os
import torch

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def save_model(self, model_path):
        
        encoder_path = os.path.join(model_path, "encoder_latest.pth")
        torch.save(self.encoder.state_dict(), encoder_path)

        decoder_path = os.path.join(model_path, "decoder_latest.pth")
        torch.save(self.decoder.state_dict(), decoder_path)

        latent_path = os.path.join(model_path, "latent_latest.pth")
        torch.save(self.latent.state_dict(), latent_path)


    def load_model(self, model_path):
        encoder_path = os.path.join(model_path, "encoder_latest.pth")
        decoder_path = os.path.join(model_path, "decoder_latest.pth")
        latent_path = os.path.join(model_path, "latent_latest.pth")

        self.encoder.eval()
        self.decoder.eval()
        self.latent.eval()

        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.latent.load_state_dict(torch.load(latent_path))
