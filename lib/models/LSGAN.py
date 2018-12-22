import torch
from torch import nn
from ..BaseModels import BaseGenerator, BaseDiscriminator, weights_init
from .GAN import GAN


class DiscriminatorLS(BaseDiscriminator):
    def __init__(self, n_attrs, a=-1, b=1, c=0, n_fil=32):
        super(DiscriminatorLS, self).__init__(n_attrs, n_fil,
                                              norm_layer=nn.BatchNorm2d,
                                              out_f=None)
        self.a = a
        self.b = b
        self.c = c
        self.apply(weights_init)

    def G_loss(self, fake_img, fake_attrs, losses):
        f_score = self(fake_img, fake_attrs)
        g_loss = 0.5 * torch.mean((f_score - self.c) ** 2)
        losses["g_loss"].append(g_loss.detach().cpu().numpy())
        return g_loss

    def D_loss(self, real_img, real_attrs, fake_img, fake_attrs, losses=None):
        r_score = self(real_img, real_attrs)
        f_score = self(fake_img.detach(), fake_attrs)

        d_loss = 0.5 * torch.mean((r_score - self.a) ** 2) + 0.5 * torch.mean((f_score - self.b) ** 2)

        if losses is not None:
            losses["d_loss"].append(d_loss.detach().cpu().numpy())
        return d_loss


class LSGAN(GAN):
    def __init__(self, n_fil, z_dim, n_attrs, viz):
        super(LSGAN, self).__init__(viz)
        self.G = BaseGenerator(z_dim, n_attrs, n_fil=n_fil)
        self.D = DiscriminatorLS(n_attrs, n_fil=n_fil)
