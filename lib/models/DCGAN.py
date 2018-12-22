import torch
from torch import nn
from ..BaseModels import BaseGenerator, BaseDiscriminator, weights_init
from .GAN import GAN


class Discriminator(BaseDiscriminator):
    def __init__(self, n_attrs, n_fil=32):
        super(Discriminator, self).__init__(n_attrs, n_fil,
                                            norm_layer=nn.BatchNorm2d,
                                            out_f=torch.sigmoid)
        self.apply(weights_init)

    def G_loss(self, fake_img, fake_attrs, losses):
        f_prob = self(fake_img, fake_attrs)
        g_loss = -torch.log(f_prob + 1e-8).mean()
#         g_loss = torch.log(1 - f_prob + 1e-8).mean()

        if losses is not None:
            losses["g_loss"].append(g_loss.detach().cpu().numpy())
        return g_loss

    def D_loss(self, real_img, real_attrs, fake_img, fake_attrs, losses=None):
        r_prob = self(real_img, real_attrs)
        f_prob = self(fake_img.detach(), fake_attrs)
        log_r_prob = torch.log(r_prob + 1e-8).mean()
        log_f_prob = torch.log(1 - f_prob + 1e-8).mean()
        d_loss = -(log_r_prob + log_f_prob)

        if losses is not None:
            losses["d_loss"].append(d_loss.detach().cpu().numpy())
            losses["r_prob"].append(r_prob.mean().detach().cpu().numpy())
            losses["f_prob"].append(f_prob.mean().detach().cpu().numpy())
        return d_loss


class DCGAN(GAN):
    def __init__(self, n_fil, z_dim, n_attrs, viz):
        super(DCGAN, self).__init__(viz)
        self.G = BaseGenerator(z_dim, n_attrs, n_fil=n_fil)
        self.D = Discriminator(n_attrs, n_fil=n_fil)
