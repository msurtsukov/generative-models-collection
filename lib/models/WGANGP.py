import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from ..BaseModels import BaseGenerator
from .GAN import GAN
from .WGAN import DiscriminatorW


class DiscriminatorWGP(DiscriminatorW):
    def __init__(self, n_attrs, n_fil=32, lmbda=10):
        super(DiscriminatorW, self).__init__(n_attrs, n_fil,
                                             norm_layer=nn.InstanceNorm2d,
                                             out_f=None)
        self.lmbda = lmbda

    def calc_gradient_penalty(self, real_img, fake_img, attrs):
        batch_size, c, w, h = real_img.shape
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, real_img.nelement(
        ) // batch_size).contiguous().view(batch_size, c, w, h)
        alpha = alpha.cuda() if real_img.is_cuda else alpha

        interpolates = alpha * real_img + ((1 - alpha) * fake_img)

        if real_img.is_cuda:
            interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self(interpolates, attrs)
        grad_outputs = torch.ones_like(disc_interpolates)
        if real_img.is_cuda:
            grad_outputs = grad_outputs.cuda()
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lmbda
        return gradient_penalty

    def D_loss(self, real_img, real_attrs, fake_img, fake_attrs, losses=None):
        r_score = self(real_img, real_attrs)
        f_score = self(fake_img.detach(), fake_attrs)

        wd = r_score.mean() - f_score.mean()  # Wasserstein-1 Distance
        gp = self.calc_gradient_penalty(real_img.data, fake_img.data, real_attrs)

        d_loss = gp - wd

        if losses is not None:
            losses["wd"].append(wd.detach().cpu().numpy())
            losses["gp"].append(gp.detach().cpu().numpy())
            losses["d_loss"].append(d_loss.detach().cpu().numpy())
        return d_loss


class WGANGP(GAN):
    def __init__(self, n_fil, z_dim, n_attrs, viz):
        super(WGANGP, self).__init__(viz)
        self.G = BaseGenerator(z_dim, n_attrs, n_fil=n_fil)
        self.D = DiscriminatorWGP(n_attrs, n_fil=n_fil)
        self.wasserstein = True
