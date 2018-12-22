import torch
from torch import nn
import torch.nn.functional as F
from ..BaseModels import BaseGenerator, Encoder
from .GAN import GAN


class DiscriminatorAE(nn.Module):
    def __init__(self, n_attrs, z_dim, n_fil, gamma, lmbda):
        super(DiscriminatorAE, self).__init__()
        self.encoder = Encoder(n_attrs, z_dim, n_fil)
        self.decoder = BaseGenerator(z_dim, n_attrs, n_fil)
        self.gamma = gamma
        self.lmbda = lmbda

        self.k = 0
        self.g_loss_last = 0

    def ae_loss(self, img, attrs):
        z = self.encoder(img, attrs)
        decoded = self.decoder(z, attrs)
        loss = torch.mean(torch.abs(decoded-img))
        return loss

    def G_loss(self, fake_img, fake_attrs, losses):
        g_loss = self.ae_loss(fake_img, fake_attrs)
        self.g_loss_last = g_loss.detach()
        losses["g_loss"].append(self.g_loss_last.cpu().numpy())
        return g_loss

    def D_loss(self, real_img, real_attrs, fake_img, fake_attrs, losses=None):
        r_loss = self.ae_loss(real_img, real_attrs)
        f_loss = self.ae_loss(fake_img.detach(), fake_attrs)

        d_loss = r_loss - f_loss * self.k

        gamma_r_loss = self.gamma*r_loss
        self.k = self.k + self.lmbda * (gamma_r_loss - self.g_loss_last).detach().cpu().numpy()
        self.k = min(1., max(0., self.k))
        m_global = r_loss + torch.abs(gamma_r_loss - self.g_loss_last)
        if losses is not None:
            losses["r_loss"].append(r_loss.detach().cpu().numpy())
            losses["f_loss"].append(f_loss.detach().cpu().numpy())
            losses["d_loss"].append(d_loss.detach().cpu().numpy())
            losses["k"].append(self.k)
            losses["m_global"].append(m_global.detach().cpu().numpy())
        return d_loss


class BEGAN(GAN):
    def __init__(self, n_fil, z_dim, n_attrs, viz):
        super(BEGAN, self).__init__(viz)
        self.G = BaseGenerator(z_dim, n_attrs, n_fil)
        self.D = DiscriminatorAE(n_attrs, z_dim, n_fil, gamma=0.5, lmbda=0.001)
