import torch
from torch import nn
import torch.nn.functional as F
from ..BaseModels import BaseGenerator, BaseDiscriminator, weights_init
from .GAN import GAN


class QInfo(nn.Module):
    def __init__(self, n_attrs, n_fil=32):
        super(QInfo, self).__init__()
        self.n_attrs = n_attrs
        self.linear_1 = nn.Linear(n_fil * 128, n_fil)
        self.batch_lin_n = nn.BatchNorm1d(n_fil)
        self.linear_out = nn.Linear(n_fil, self.n_attrs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.batch_lin_n(self.linear_1(x)))
        logits = self.linear_out(x)
        return logits


class DiscriminatorInfo(BaseDiscriminator):
    """
    This discriminator considers hidden attributes to be binomial
    """

    def __init__(self, n_attrs, lmbda=1, n_fil=32):
        super(DiscriminatorInfo, self).__init__(0, n_fil,
                                                norm_layer=nn.BatchNorm2d,
                                                out_f=torch.sigmoid)
        self.Q = QInfo(n_attrs, n_fil)
        self.lmbda = lmbda
        self.apply(weights_init)

    def head(self, img):
        x = 2. * img - 1.

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_n(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_n(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_n(self.conv4(x)), 0.2)
        return x

    def forward(self, img, attrs=None):
        x = self.head(img)
        x = self.out_f(self.conv5(x))
        return x

    @staticmethod
    def calculate_mutual_info_loss(attrs, pred_attrs_logits):
        # convert attrs to binomial
        # no prior entropy part here as is it independent of models parameters
        # may be also optimized in case prior is not explicitly provided
        return F.binary_cross_entropy_with_logits(pred_attrs_logits, attrs)

    def G_loss(self, fake_img, fake_attrs, losses):
        x = self.head(fake_img)
        f_prob = self.out_f(self.conv5(x))  # probability output
        attrs_logits = self.Q(x)  # logits of attributes probabilities

        cross_ent = self.calculate_mutual_info_loss(fake_attrs, attrs_logits)
        g_loss = -torch.log(f_prob + 1e-8).mean()  # it seems wrong to use this trick here as together with
        # cross entropy loss loses right probabilistic interpretation, however this way it work much better

        # g_loss = torch.log(1 - f_prob + 1e-8).mean()

        if losses is not None:
            losses["g_loss"].append(g_loss.detach().cpu().numpy())

        return g_loss + cross_ent * self.lmbda

    def D_loss(self, real_img, real_attrs, fake_img, fake_attrs, losses=None):
        r_prob = self(real_img, None)

        x = self.head(fake_img.detach())
        f_prob = self.out_f(self.conv5(x))
        f_attrs_logits = self.Q(x)

        log_r_prob = torch.log(r_prob + 1e-8)
        log_f_prob = torch.log(1 - f_prob + 1e-8)
        d_loss = -(log_r_prob + log_f_prob).mean()

        cross_ent = self.calculate_mutual_info_loss(fake_attrs, f_attrs_logits)

        if losses is not None:
            losses["d_loss"].append(d_loss.detach().cpu().numpy())
            losses["r_prob"].append(r_prob.mean().detach().cpu().numpy())
            losses["f_prob"].append(f_prob.mean().detach().cpu().numpy())
            losses["cross_ent"].append(cross_ent.detach().cpu().numpy())

        return d_loss + cross_ent * self.lmbda


class InfoGAN(GAN):
    def __init__(self, n_fil, z_dim, n_attrs, viz):
        super(InfoGAN, self).__init__(viz)
        self.G = BaseGenerator(z_dim, n_attrs, n_fil=n_fil)
        self.D = DiscriminatorInfo(n_attrs, n_fil=n_fil)

    def init_optimizers(self, lr, beta1, beta2):
        super(InfoGAN, self).init_optimizers(lr, beta1, beta2)
        # here we make sure that during G step mutual info loss is also minimized
        self.g_optim = torch.optim.Adam([{'params': self.G.parameters()}, {'params': self.D.Q.parameters()}],
                                        lr=lr, betas=(beta1, beta2))
