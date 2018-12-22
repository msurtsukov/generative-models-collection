from torch import nn
from ..BaseModels import BaseGenerator, BaseDiscriminator
from .GAN import GAN


class DiscriminatorW(BaseDiscriminator):
    def __init__(self, n_attrs, n_fil=32):
        super(DiscriminatorW, self).__init__(n_attrs, n_fil,
                                             norm_layer=nn.BatchNorm2d,
                                             out_f=None)

    def clip_parameters(self, clip_value):
        for p in self.parameters():
            p.data = p.clamp(-clip_value, clip_value).data

    def G_loss(self, fake_img, fake_attrs, losses):
        f_score = self(fake_img, fake_attrs)
        g_loss = -f_score.mean()
        losses["g_loss"].append(g_loss.detach().cpu().numpy())
        return g_loss

    def D_loss(self, real_img, real_attrs, fake_img, fake_attrs, losses=None):
        self.clip_parameters(0.01)
        r_score = self(real_img, real_attrs)
        f_score = self(fake_img.detach(), fake_attrs)

        wd = r_score.mean() - f_score.mean()  # Wasserstein-1 Distance

        d_loss = -wd
        if losses is not None:
            losses["d_loss"].append(d_loss.detach().cpu().numpy())
        return d_loss


class WGAN(GAN):
    def __init__(self, n_fil, z_dim, n_attrs, viz):
        super(WGAN, self).__init__(viz)
        self.G = BaseGenerator(z_dim, n_attrs, n_fil=n_fil)
        self.D = DiscriminatorW(n_attrs, n_fil=n_fil)
        self.wasserstein = True
