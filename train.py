import os, sys
import argparse

import torch
import visdom
from lib.utils import CroppedCelebA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='BEGAN',
                        choices=['VAE', 'DCGAN', 'VAEGAN', 'LSGAN', 'WGAN', 'WGANGP', "InfoGAN", 'BEGAN'],
                        help='The type of GAN: VAE, DCGAN, VAEGAN, LSGAN, WGAN, WGANGP, InfoGAN, BEGAN')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch of preloaded model')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Training learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')

    parser.add_argument('--n_fil', type=int, default=128, help='Min number of filters of conv/deconv layers')
    parser.add_argument('--z_dim', type=int, default=88, help='Dimension of latent space')

    parser.add_argument('--n_critic', type=int, default=5,
                        help='Number of discriminator steps per generator step. Applicable for Wassertein models')
    parser.add_argument('--n_gen', type=int, default=1,
                        help='Number of generator steps per discriminator step. Applicable for non-Wassertein models')

    parser.add_argument('--use_visdom', type=bool, default=True, help='Use Visdom')
    parser.add_argument('--visdom_host', type=str, default='http://localhost', help='Visdom host')
    parser.add_argument('--visdom_port', type=int, default=8889, help='Visdom port')
    parser.add_argument('--visdom_env', type=str, default='main', help='Visdom environment')
    parser.add_argument('--plot_freq', type=int, default=100, help='Visdom plotting frequency')

    parser.add_argument('--preload_model', type=bool, default=False,
                        help='Continue training from the model specified by load_dir and start_epoch')
    parser.add_argument('--load_dir', type=str, default='models',
                        help='Parent directory path to preload the model from')
    parser.add_argument('--save_dir', type=str, default='models', help='Parent directory path to save the model')
    parser.add_argument('--chkp_freq', type=int, default=20, help='Model checkpoint frequency')
    parser.add_argument('--imgs_dir', type=str, default='results',
                        help='Parent directory path to save the generated images')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(43)

    if args.type == "VAE":
        from lib.models.VAE import VAE

        Model = VAE
    elif args.type == "VAEGAN":
        from lib.models.VAEGAN import VAEGAN

        Model = VAEGAN
    elif args.type == "DCGAN":
        from lib.models.DCGAN import DCGAN

        Model = DCGAN
    elif args.type == "LSGAN":
        from lib.models.LSGAN import LSGAN

        Model = LSGAN
    elif args.type == "WGAN":
        from lib.models.WGAN import WGAN

        Model = WGAN
    elif args.type == "WGANGP":
        from lib.models.WGANGP import WGANGP

        Model = WGANGP
    elif args.type == "InfoGAN":
        from lib.models.InfoGAN import InfoGAN

        Model = InfoGAN
    elif args.type == "BEGAN":
        from lib.models.BEGAN import BEGAN

        Model = BEGAN
    else:
        print("Wrong GAN type provided")
        sys.exit()

    celeba_data = CroppedCelebA("./data")
    n_attrs = 40

    # enforce directories existence
    imgs_dir = os.path.join(args.imgs_dir, args.type)
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)

    save_dir = os.path.join(args.save_dir, args.type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    load_dir = os.path.join(args.load_dir, args.type)

    if args.use_visdom:
        viz = visdom.Visdom(server=args.visdom_host, port=args.visdom_port, env=args.visdom_env)
    else:
        viz = None

    model = Model(args.n_fil, args.z_dim, n_attrs, viz).cuda()
    model.init_optimizers(args.lr, args.beta1, args.beta2)
    model.n_gen = args.n_gen
    model.n_critic = args.n_critic
    if args.preload_model:
        model.load(load_dir, args.start_epoch-1)

    try:
        model.train_model(celeba_data, n_epochs=args.n_epochs, batch_size=args.batch_size,
                          start_epoch=args.start_epoch,
                          models_dir=save_dir, imgs_dir=imgs_dir,
                          chkp_freq=args.chkp_freq, plot_freq=args.plot_freq)
    except KeyboardInterrupt:
        pass
    finally:
        # doesn't work for now when keyboard interrupted, some problems with file writing
        model.save(save_dir)
