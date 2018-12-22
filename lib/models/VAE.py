import torch
from torch import nn
import torch.nn.functional as F
from ..BaseModels import BaseGenerator, Encoder, BaseModel


class VAE(BaseModel):
    def __init__(self, n_fil, z_dim, n_attrs, viz):
        super(VAE, self).__init__(viz)
        self.encoder = Encoder(n_attrs, 2*z_dim, n_fil)
        self.G = BaseGenerator(z_dim, n_attrs, n_fil)
        self.optim = None

    def init_optimizers(self, lr, beta1, beta2):
        self.base_lr = lr
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizers = [self.optim]

    def kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var))
        return kl_loss

    def sample(self, z_mean, z_log_var):
        std = torch.exp(z_log_var) / 2.
        return torch.randn(*z_mean.size()).cuda() * std + z_mean

    def train_step(self, step, img, attrs):
        bs, ch, h, w = img.size()
        elements_per_img = ch*h*w
        x = self.encoder(img, attrs)
        z_mean = x[:, :self.G.z_dim]
        z_log_var = x[:, self.G.z_dim:]

        z = self.sample(z_mean, z_log_var)
        decoded_img = self.G(z, attrs)
        xent_loss = F.binary_cross_entropy(decoded_img, img, reduction="sum") / elements_per_img
        kl_loss = self.kl_loss(z_mean, z_log_var) / elements_per_img
        loss = xent_loss + kl_loss
        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        self.optim.step()

        self.temp_losses["xent_loss"].append(xent_loss.detach().cpu().numpy())
        self.temp_losses["kl_loss"].append(kl_loss.detach().cpu().numpy())

        self.scatters["z_mean"] = z_mean[:, :2].detach().cpu().numpy()
        self.images["real"] = img[:32]
        self.images["decoded"] = decoded_img[:32]
        return z, attrs
