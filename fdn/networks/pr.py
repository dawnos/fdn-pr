
import torch.nn as nn
import torch.utils.model_zoo
import torch

from ..utils import Conv2dBlock
from ..base_net import BaseFeatureDisentanglementNetwork


class PlaceRecognitionFDN(BaseFeatureDisentanglementNetwork):
  def __init__(self, args):
    # Sub-networks
    super().__init__(args)

  @staticmethod
  def create_place_encoder(args):
    return PlaceEncoder()

  @staticmethod
  def create_appearance_encoder(args):
    return AppearanceEncoder()

  @staticmethod
  def create_decoder(args):
    return Decoder()

  @staticmethod
  def create_place_domain_discriminator(args):
    return PlaceDomainDiscriminator()

  @staticmethod
  def create_appearance_compatibility_discriminator(args):
    return AppearanceCompatibilityDiscriminator()

  # def forward(self, data1, data2, args):
  #
  #   # Extract features
  #   code_l1, code_g1 = self.encode(data1)
  #   code_l2, code_g2 = self.encode(data2)
  #
  #   # Reconsutruction from decoder
  #   recon1 = self.decode(code_l1, code_g1)
  #   recon2 = self.decode(code_l2, code_g2)
  #
  #   return code_l1, code_g1, code_l2, code_g2, recon1, recon2
  #
  # def encode(self, x):
  #   code_l = self.lenc(x)
  #   code_g = self.genc(x)
  #   return code_l, code_g
  #
  # def decode(self, code_l, code_g):
  #   images = self.gen(code_l, code_g)
  #   return images


class PlaceEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      Conv2dBlock(3, 64, 7, 1, 3, 'in', 'relu', 'reflect'),
      Conv2dBlock(64, 64, 4, 2, 1, 'in', 'relu', 'reflect'),
      Conv2dBlock(64, 128, 4, 2, 1, 'in', 'relu', 'reflect'),
      Conv2dBlock(128, 128, 4, 2, 1, 'in', 'relu', 'reflect'),
      Conv2dBlock(128, 128, 4, 2, 1, 'in', 'relu', 'reflect'),
      Conv2dBlock(128, 64, 3, 1, 1, 'in', 'relu', 'reflect'),
    )

  def forward(self, x):
    return self.net(x)


class AppearanceEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.features = nn.Sequential(
      Conv2dBlock(3, 64, 7, 1, 3, 'none', 'relu', 'reflect'),
      Conv2dBlock(64, 64, 4, 2, 1, 'none', 'relu', 'reflect'),
      Conv2dBlock(64, 128, 4, 2, 1, 'none', 'relu', 'reflect'),
      Conv2dBlock(128, 256, 4, 2, 1, 'none', 'relu', 'reflect'),
      Conv2dBlock(256, 512, 4, 2, 1, 'none', 'relu', 'reflect'),
      nn.AdaptiveAvgPool2d([1, 1]),
      Conv2dBlock(512, 8, 1, 1, 0, 'none', 'none'),
    )
    self.linear = nn.Sequential(
      # nn.Linear(1*1*512, 64),
      # nn.LeakyReLU(0.2),
      # nn.Linear(64, 8),
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.linear(x)
    x = x.view(x.size(0), 8, 1, 1)
    return x


class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(64 + 8, 128, 1, padding=0),
      nn.Upsample(scale_factor=2),
      Conv2dBlock(128, 128, 5, 1, 2, 'ln', 'relu', 'reflect'),
      nn.Upsample(scale_factor=2),
      Conv2dBlock(128, 128, 5, 1, 2, 'ln', 'relu', 'reflect'),
      nn.Upsample(scale_factor=2),
      Conv2dBlock(128, 64, 5, 1, 2, 'ln', 'relu', 'reflect'),
      nn.Upsample(scale_factor=2),
      Conv2dBlock(64, 3, 7, 1, 3, 'none', 'tanh', 'reflect'),
      # nn.Tanh()
    )

  def forward(self, xf, xg):
    xg2 = xg.repeat([1, 1, xf.size(2), xf.size(3)])
    x = torch.cat((xf, xg2), dim=1)
    return self.net(x)


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.pool1 = nn.Sequential()
    self.pool2 = nn.Sequential()
    self.net = nn.Sequential()
    self.linear = nn.Sequential()

  def forward(self, x1, x2):

    x1 = self.pool1(x1)
    x2 = self.pool2(x2)
    if x1.shape != x2.shape:
      x2 = x2.repeat([1, 1, x1.shape[2], x1.shape[3]])
    x = torch.cat((x1, x2), dim=1)
    y = self.predict(x)
    return y, x

  def predict(self, x):
    x = self.net(x)
    x_fat = x.view(x.size(0), -1)
    y = self.linear(x_fat)
    return y


class AppearanceCompatibilityDiscriminator(Discriminator):
  def __init__(self):
    super().__init__()

    self.net = nn.Sequential(
      Conv2dBlock(64 + 8, 128, 4, 2, 1, 'ln', 'lrelu', 'reflect'),
      Conv2dBlock(128, 256, 4, 2, 1, 'ln', 'lrelu', 'reflect'),
      Conv2dBlock(256, 512, 4, 2, 1, 'ln', 'lrelu', 'reflect'),
      Conv2dBlock(512, 64, 1, 1, 0, 'none', 'lrelu'),
      Conv2dBlock(64, 1, 1, 1, 0, 'none', 'none'),
    )
    self.linear = nn.Sequential(
      # nn.Linear(1 * 1 * 512, 128),
      # nn.LeakyReLU(0.2),
      # nn.Linear(128, 1),
    )


class PlaceDomainDiscriminator(Discriminator):
  def __init__(self):
    super().__init__()

    self.net = nn.Sequential(
      Conv2dBlock(64 + 64, 256, 4, 2, 1, 'ln', 'lrelu', 'reflect'),  #
      Conv2dBlock(256, 512, 4, 2, 1, 'ln', 'lrelu', 'reflect'),
      Conv2dBlock(512, 1024, 4, 2, 1, 'ln', 'lrelu', 'reflect'),
      Conv2dBlock(1024, 128, 1, 1, 0, 'none', 'lrelu'),
      Conv2dBlock(128, 1, 1, 1, 0, 'none', 'none'),
    )
    self.linear = nn.Sequential(
      # nn.Linear(1 * 1 * 1024, 256),
      # nn.LeakyReLU(0.2),
      # nn.Linear(256, 1),
    )
