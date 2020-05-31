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


class MNISTFDN(BaseFeatureDisentanglementNetwork):

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


class PlaceEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(3, 8, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d([2, 2]),
      nn.Conv2d(8, 16, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d([2, 2]),
      nn.Conv2d(16, 32, 3, padding=2),
      nn.ReLU(),
      nn.MaxPool2d([2, 2]),
      nn.Conv2d(32, 64, 3, padding=1),
    )

  def forward(self, x):
    return self.net(x)


class AppearanceEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.features = nn.Sequential(
      # conv1
      nn.Conv2d(3, 8, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d([2, 2]),
      # conv2
      nn.Conv2d(8, 16, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d([2, 2]),
      # conv3
      nn.Conv2d(16, 32, 3, padding=2),
      nn.ReLU(),
      nn.MaxPool2d([2, 2]),
    )
    self.conv1 = nn.Sequential(
      # conv4
      nn.Conv2d(32, 32, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d([2, 2]),
      # conv5
      nn.Conv2d(32, 16, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d([2, 2]),
      # conv6
      nn.Conv2d(16, 8, 3, padding=1),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.conv1(x)
    return x


class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(64 + 8, 128, 1, padding=0),
      nn.Upsample(size=7),
      nn.Conv2d(128, 32, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 16, 3, padding=1),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(16, 16, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 8, 3, padding=1),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(8, 8, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(8, 3, 3, padding=1),
    )

  def forward(self, xf, xg):
    xg2 = xg.repeat([1, 1, xf.size(2), xf.size(3)])
    x = torch.cat((xf, xg2), dim=1)
    return self.net(x)


class AppearanceCompatibilityDiscriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(64 + 8, 32, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 16, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 8, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(8, 4, 3, padding=1),
    )
    self.classifier = nn.Sequential(
      nn.Linear(64, 16),
      nn.Linear(16, 1),
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
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(64 + 64, 64, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 32, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 16, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 8, 3, padding=1),
    )
    self.classifier = nn.Sequential(
      nn.Linear(128, 32),
      nn.Linear(32, 8),
      nn.Linear(8, 1),
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
