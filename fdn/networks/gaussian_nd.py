"""
Copyright (c) 2019 Li Tang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch.nn as nn
import torch.utils.model_zoo
import torch
from ..base_net import BaseFeatureDisentanglementNetwork


class GaussianNDFDN(BaseFeatureDisentanglementNetwork):

  def __init__(self, args):
    super().__init__(args)

  @staticmethod
  def create_place_encoder(args):
    return PlaceEncoder(args)

  @staticmethod
  def create_appearance_encoder(args):
    return AppearanceEncoder(args)

  @staticmethod
  def create_decoder(args):
    return Decoder(args)

  @staticmethod
  def create_place_domain_discriminator(args):
    return PlaceDomainDiscriminator(args)

  @staticmethod
  def create_appearance_compatibility_discriminator(args):
    return AppearanceCompatibilityDiscriminator(args)


class PlaceEncoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    dim_s = args.dim_enc_s  # len(args.mean_s)
    dim_x = args.dim_x
    self.net = nn.Sequential(
      nn.Conv2d(dim_x, dim_s, 1),
    )

  def forward(self, x):
    return self.net(x)


class AppearanceEncoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    dim_a = args.dim_enc_a  # len(args.mean_a)
    dim_x = args.dim_x
    self.features = nn.Sequential(
      nn.Conv2d(dim_x, dim_a, 1),
    )
    self.conv1 = nn.Sequential(
    )

  def forward(self, x):
    x = self.features(x)
    x = self.conv1(x)
    return x


class Decoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    dim_s = args.dim_enc_s  # len(args.mean_s)
    dim_a = args.dim_enc_a  # len(args.mean_a)
    dim_x = args.dim_x
    self.net = nn.Sequential(
      nn.Conv2d(dim_s + dim_a, dim_x, 1),
    )

  def forward(self, xf, xg):
    xg2 = xg.repeat([1, 1, xf.size(2), xf.size(3)])
    x = torch.cat((xf, xg2), dim=1)
    return self.net(x)


class AppearanceCompatibilityDiscriminator(nn.Module):
  def __init__(self, args):
    super().__init__()
    dim_s = args.dim_enc_s  # len(args.mean_s)
    dim_a = args.dim_enc_a  # len(args.mean_a)
    dim_hidden = (dim_s + dim_a) * 4
    self.net = nn.Sequential(
      nn.Conv2d(dim_s + dim_a, dim_hidden, 1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(dim_hidden, dim_hidden, 1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(dim_hidden, dim_hidden, 1),
    )
    self.classifier = nn.Sequential(
      nn.Linear(dim_hidden, 1),
    )

  def forward(self, xf, xg):
    xg2 = xg.repeat([1, 1, xf.size(2), xf.size(3)])
    x = torch.cat((xf, xg2), dim=1)
    y = self.predict(x)

    return y, x

  def predict(self, x):
    x = self.net(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x


class PlaceDomainDiscriminator(nn.Module):
  def __init__(self, args):
    super().__init__()
    dim_s = args.dim_enc_s  # len(args.mean_s)
    dim_hidden = dim_s * 2 * 4
    self.net = nn.Sequential(
      nn.Conv2d(dim_s * 2, dim_hidden, 1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(dim_hidden, dim_hidden, 1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(dim_hidden, dim_hidden, 1),
    )
    self.classifier = nn.Sequential(
      nn.Linear(dim_hidden, 1),
    )

  def forward(self, xf, xg):
    x = torch.cat((xf, xg), dim=1)
    y = self.predict(x)

    return y, x

  def predict(self, x):
    x = self.net(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
