import torch
from torch import nn
import torch.nn.functional as F
from ..BaseModels import BaseGenerator, Encoder, BaseModel, BaseDiscriminator, weights_init


class DiscriminatorVAEGAN(BaseDiscriminator):
    def __init__(self, n_attrs, n_fil):
        super(DiscriminatorVAEGAN, self).__init__(n_attrs, n_fil, out_f=torch.sigmoid)
        
    def head(self, img, attrs):
        x = 2. * img - 1.

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_n(self.conv2(x)), 0.2)
        # append attributes to channels
        if self.n_attrs:
            batch_size, c, w, h = x.shape
            expanded_attrs = attrs.view(
                *attrs.shape, 1, 1).expand(*attrs.shape, w, h)
            x = torch.cat((x, expanded_attrs), dim=1)

        x = self.conv3_n(self.conv3(x))
        x = F.leaky_relu(x, 0.2)
        # append attributes to channels
        if self.n_attrs:
            batch_size, c, w, h = x.shape
            expanded_attrs = attrs.view(
                *attrs.shape, 1, 1).expand(*attrs.shape, w, h)
            x = torch.cat((x, expanded_attrs), dim=1)

        x = self.conv4_n(self.conv4(x))
        return x

    def tail(self, x, attrs):
        x = F.leaky_relu(x, 0.2)
        x = self.out_f(self.conv5(x)) if self.out_f else self.conv5(x)
        return x

    def forward(self, img, attrs):
        l = self.head(img, attrs)
        x = self.tail(l, attrs)
        return l, x


class VAEGAN(BaseModel):
    def __init__(self, n_fil, z_dim, n_attrs, viz, gamma=0.01):
        super(VAEGAN, self).__init__(viz)
        self.E = Encoder(n_attrs, 2*z_dim, n_fil)
        self.G = BaseGenerator(z_dim, n_attrs, n_fil)
        self.D = DiscriminatorVAEGAN(n_attrs, n_fil)

        self.z_dim = z_dim
        self.gamma = gamma

        self.optim_e = None
        self.optim_g = None
        self.optim_d = None

    def init_optimizers(self, lr, beta1, beta2):
        self.base_lr = lr
        self.optim_e = torch.optim.Adam(self.E.parameters(), lr=lr, betas=(beta1, beta2))
        self.optim_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.optim_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizers = [self.optim_e, self.optim_g, self.optim_d]

    def kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var))
        return kl_loss

    def sample(self, z_mean, z_log_var):
        std = torch.exp(z_log_var) / 2.
        return torch.randn(*z_mean.size()).cuda() * std + z_mean

    def disllike_loss(self, l_decoded, l_real):
        # loss = torch.sum()
        loss = F.mse_loss(l_decoded, l_real, reduction="sum")
        return loss

    def train_step(self, step, img, attrs):
        bs, ch, h, w = img.size()
        elements_per_img = ch*h*w

        x = self.E(img, attrs)
        z_mean_x = x[:, :self.G.z_dim]
        z_log_var_x = x[:, self.G.z_dim:]
        z_x = self.sample(z_mean_x, z_log_var_x)
        kl_loss = self.kl_loss(z_mean_x, z_log_var_x) / elements_per_img
        decoded_img = self.G(z_x, attrs)

        z = torch.randn(bs, self.z_dim).cuda()
        fake_img = self.G(z, attrs)

        r_l, r_prob = self.D(img, attrs)
        d_l, d_prob = self.D(decoded_img, attrs)
        f_l, f_prob = self.D(fake_img, attrs)

        # compute losses
        # encoder loss
        dll_loss = self.disllike_loss(r_l, d_l) / elements_per_img
        e_loss = dll_loss + kl_loss

        # generator/decoder loss
        # g_loss = (torch.log(1 - f_prob + 1e-8).mean() + torch.log(1 - d_prob + 1e-8).mean())  # so it was in the paper
        g_loss = -(torch.log(f_prob + 1e-8).mean() + torch.log(d_prob + 1e-8).mean())
        g_loss = g_loss + dll_loss * self.gamma

        # discriminator loss
        log_r_prob = torch.log(r_prob + 1e-8).mean()
        log_d_prob = torch.log(1 - d_prob + 1e-8).mean()
        log_f_prob = torch.log(1 - f_prob + 1e-8).mean()
        d_loss = -(log_r_prob + log_f_prob + log_d_prob)

        self.optim_e.zero_grad()
        e_loss.backward(retain_graph=True)
        self.optim_e.step()

        self.optim_g.zero_grad()
        g_loss.backward(retain_graph=True)
        self.optim_g.step()

        if step % self.n_gen == 0:
            self.optim_d.zero_grad()
            d_loss.backward(retain_graph=True)
            self.optim_d.step()

        self.temp_losses["kl_loss"].append(float(kl_loss.detach().cpu().numpy()))
        self.temp_losses["dll_loss"].append(float(dll_loss.detach().cpu().numpy()))

        self.temp_losses["r_prob"].append(float(r_prob.mean().detach().cpu().numpy()))
        self.temp_losses["d_prob"].append(float(d_prob.mean().detach().cpu().numpy()))
        self.temp_losses["f_prob"].append(float(f_prob.mean().detach().cpu().numpy()))

        self.temp_losses["e_loss"].append(float(e_loss.detach().cpu().numpy()))
        self.temp_losses["g_loss"].append(float(g_loss.detach().cpu().numpy()))
        self.temp_losses["d_loss"].append(float(d_loss.detach().cpu().numpy()))

        self.images["real"] = img[:32]
        self.images["decoded"] = decoded_img[:32]
        return z, attrs
