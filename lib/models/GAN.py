import torch
from torch import nn
from torch.autograd import Variable
from ..BaseModels import BaseModel


class GAN(BaseModel):
    def __init__(self, viz):
        super(GAN, self).__init__(viz)
        self.wasserstein = False
        self.G = None
        self.D = None
        self.g_optim = None
        self.d_optim = None

    def gen_step(self, z, attrs):
        f_img = self.G(z, attrs)
        g_loss = self.D.G_loss(f_img, attrs, self.temp_losses)

        self.G.zero_grad()
        g_loss.backward(retain_graph=True)
        self.g_optim.step()

    def dis_step(self, real_img, real_attrs, z):
        f_img = self.G(z, real_attrs).detach()
        d_loss = self.D.D_loss(real_img, real_attrs, f_img, real_attrs, self.temp_losses)
        self.D.zero_grad()
        d_loss.backward(retain_graph=True)
        self.d_optim.step()

    def init_optimizers(self, lr, beta1, beta2):
        self.base_lr = lr
        self.g_optim = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.d_optim = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizers = [self.g_optim, self.d_optim]

    def train_step(self, step, img, attrs):
        bs = img.size(0)

        # D training
        z = Variable(torch.randn(bs, self.G.z_dim)).cuda()
        self.dis_step(img, attrs, z)

        # train G
        # for Wasserstein GANs we have multiple d_steps per one g_step
        # for not Wasserstein GANs it is the other way around
        if self.wasserstein:
            if step % self.n_critic == 0:
                z, attrs = self.get_random_z_attrs(bs)
                self.gen_step(z, attrs)
        else:
            for _ in range(self.n_gen):
                z, attrs = self.get_random_z_attrs(bs)
                self.gen_step(z, attrs)

        return z, attrs
