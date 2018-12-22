import os
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .utils import Plotter


def weights_init(m):
    """
    Custom weights initialization as suggested in DCGAN article
    :param m: module
    :return:
    """
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d, nn.Linear]:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class BaseGenerator(nn.Module):
    """
    Basic Deconvolution Generator
    """
    def __init__(self, z_dim, n_attrs, n_fil):
        """
        :param z_dim: Dimension of latent variable
        :param n_attrs: Number of conditional attributes
        :param n_fil: Number of filters in first Deconvolution
        """
        super(BaseGenerator, self).__init__()
        self.z_dim = z_dim
        self.n_z = z_dim + n_attrs
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.n_z, n_fil * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_fil * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(n_fil * 8, n_fil * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_fil * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(n_fil * 4, n_fil * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_fil * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(n_fil * 2, n_fil, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_fil),
            nn.ReLU(),

            nn.ConvTranspose2d(n_fil, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z, attrs):
        z_attrs_inp = torch.cat((z, attrs), dim=1).view(-1, self.n_z, 1, 1)
        output_img = (self.net(z_attrs_inp) + 1.0) / 2.0
        return output_img


class BaseDiscriminator(nn.Module):
    """
    Basic Convolution Discriminator
    """
    def __init__(self, n_attrs, n_fil=32,
                 norm_layer=nn.BatchNorm2d,
                 out_f=None):
        """
        :param n_attrs: Number of conditional attributes
        :param n_fil: Number of filters in first Convolution
        :param norm_layer: Which normalization layer to use
        :param out_f: Output function
        """
        super(BaseDiscriminator, self).__init__()
        self.n_attrs = n_attrs
        self.out_f = out_f
        input_channels = 3
        self.conv1 = nn.Conv2d(input_channels, n_fil, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_fil, n_fil * 2 - self.n_attrs, 4, 2, 1)
        self.conv2_n = norm_layer(n_fil * 2 - self.n_attrs)
        self.conv3 = nn.Conv2d(n_fil * 2, n_fil * 4 - self.n_attrs, 4, 2, 1)
        self.conv3_n = norm_layer(n_fil * 4 - self.n_attrs)
        self.conv4 = nn.Conv2d(n_fil * 4, n_fil * 8, 4, 2, 1)
        self.conv4_n = norm_layer(n_fil * 8)
        self.conv5 = nn.Conv2d(n_fil * 8, 1, 4, 1, 0)

    def forward(self, img, attrs):
        x = 2. * img - 1.

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_n(self.conv2(x)), 0.2)
        # append attributes to channels
        if self.n_attrs:
            batch_size, c, w, h = x.shape
            expanded_attrs = attrs.view(
                *attrs.shape, 1, 1).expand(*attrs.shape, w, h)
            x = torch.cat((x, expanded_attrs), dim=1)

        x = F.leaky_relu(self.conv3_n(self.conv3(x)), 0.2)
        # append attributes to channels
        if self.n_attrs:
            batch_size, c, w, h = x.shape
            expanded_attrs = attrs.view(
                *attrs.shape, 1, 1).expand(*attrs.shape, w, h)
            x = torch.cat((x, expanded_attrs), dim=1)

        x = F.leaky_relu(self.conv4_n(self.conv4(x)), 0.2)
        x = self.out_f(self.conv5(x)) if self.out_f else self.conv5(x)
        return x


class Encoder(BaseDiscriminator):
    def __init__(self, n_attrs, z_dim, n_fil):
        """
        :param n_attrs: Number of conditional attributes
        :param z_dim: Dimension of latent variable
        :param n_fil: Number of filters in first Convolution
        """
        super(Encoder, self).__init__(n_attrs, n_fil)
        del self.conv5

        self.linear_1 = nn.Linear(n_fil * 128, n_fil)
        self.batch_lin_n = nn.BatchNorm1d(n_fil)
        self.linear_out = nn.Linear(n_fil, z_dim)
        self.z_dim = z_dim

    def forward(self, img, attrs):
        x = 2. * img - 1.
        attrs = 2. * attrs - 1.

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_n(self.conv2(x)), 0.2)
        # append attributes to channels
        if self.n_attrs:
            batch_size, c, w, h = x.shape
            expanded_attrs = attrs.view(
                *attrs.shape, 1, 1).expand(*attrs.shape, w, h)
            x = torch.cat((x, expanded_attrs), dim=1)

        x = F.leaky_relu(self.conv3_n(self.conv3(x)), 0.2)
        # append attributes to channels
        if self.n_attrs:
            batch_size, c, w, h = x.shape
            expanded_attrs = attrs.view(
                *attrs.shape, 1, 1).expand(*attrs.shape, w, h)
            x = torch.cat((x, expanded_attrs), dim=1)

        x = F.leaky_relu(self.conv4_n(self.conv4(x)), 0.2)
        # FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.batch_lin_n(self.linear_1(x)))
        z = self.linear_out(x.view(x.size(0), -1))
        return z


class BaseModel(nn.Module, Plotter):
    def __init__(self, viz):
        nn.Module.__init__(self)
        Plotter.__init__(self, viz)
        self.celeba_data = None
        self.G = None
        self.optimizers = []
        self.z_const = None
        self.attrs_const = None
        self.base_lr = None

        self.epoch = 0

    def update_img(self, name, z, attrs):
        self.G.eval()
        f_img = self.G(z, attrs)
        self.images[name] = f_img
        self.G.train()

    def lr_scheduler(self, epoch):
        if epoch in range(20):
            lr = self.base_lr
        elif epoch in range(20, 50):
            lr = 0.2 * self.base_lr
        for opt in self.optimizers:
            opt.lr = lr

    def get_random_z_attrs(self, bs):
        z = Variable(torch.randn(bs, self.G.z_dim)).cuda()
        attrs = Variable(self.celeba_data.get_random_attrs(bs)).cuda()
        return z, attrs

    def train_model(self, celeba_data, n_epochs=100, batch_size=128, start_epoch=0,
                    models_dir=None, imgs_dir=None,
                    chkp_freq=None, plot_freq=None):
        data_loader = torch.utils.data.DataLoader(celeba_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)
        self.celeba_data = celeba_data
        # const z, attrs to draw pictures from
        if self.z_const is None:
            self.z_const = Variable(torch.randn(32, self.G.z_dim)).cuda() / 2
        if self.attrs_const is None:
            self.attrs_const = Variable(celeba_data.get_random_attrs(32)).cuda()

        for epoch in range(start_epoch, start_epoch + n_epochs):
            print(f"{epoch}:")
            self.lr_scheduler(epoch)
            self.epoch = epoch
            for i, (img, attrs) in tqdm(enumerate(data_loader), total=len(data_loader)):
                attrs = attrs * 0.5 + 0.5
                step = epoch * len(data_loader) + i + 1

                img = img.cuda()
                attrs = attrs.cuda()

                self.n += 1
                z, attrs = self.train_step(step, img, attrs)

                if self.viz is not None and (i + 1) % plot_freq == 0:
                    self.update_img("z", z[:32], attrs[:32])
                    if self.z_const is not None:
                        self.update_img("z_const", self.z_const, self.attrs_const)
                    self.plot()

            if imgs_dir:
                img_path = os.path.join(imgs_dir, str(epoch) + '.jpg')
                self.save_mpl_img(img_path, epoch, 4, 8, (16, 9), frame="z_const")

            if chkp_freq and (epoch + 1) % chkp_freq == 0:
                self.save(models_dir)

    def save(self, path_to_dir):
        path_to_dir = path_to_dir.strip("/")
        state_dict = {'model_state_dict': self.state_dict(),
                      'optimizers_state_dict': [opt.state_dict() for opt in self.optimizers],
                      'z_const': self.z_const,
                      'attrs_const': self.attrs_const,
                      'n': self.n,
                      'n_show': self.n_show,
                      'losses': self.losses}
        fn = f"{path_to_dir}/{self.__class__.__name__}_{self.epoch}.sd"
        torch.save(state_dict, fn)
        print("The model is saved: " + fn)

    def load(self, path_to_dir, start_epoch):
        state_dict = torch.load(f"{path_to_dir}/{self.__class__.__name__}_{start_epoch}.sd")
        self.load_state_dict(state_dict["model_state_dict"])
        for opt, sd in zip(self.optimizers, state_dict['optimizers_state_dict']):
            opt.load_state_dict(sd)
        self.z_const = state_dict["z_const"]
        self.attrs_const = state_dict["attrs_const"]
        self.epoch = start_epoch
        self.n = state_dict['n']
        self.n_show = state_dict['n_show']
        self.losses = state_dict['losses']

    def forward(self, z, attrs):
        return self.G(z, attrs)
