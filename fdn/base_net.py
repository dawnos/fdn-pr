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


class BaseFeatureDisentanglementNetwork(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.enc_s = self.create_place_encoder(args)
    self.enc_a = self.create_appearance_encoder(args)
    self.dec = self.create_decoder(args)
    self.dis_app = self.create_appearance_compatibility_discriminator(args)
    self.dis_pla = self.create_place_domain_discriminator(args)

  def forward(self, data1, data2, args):
    raise NotImplemented

  def encode(self, data):
    code_s = self.enc_s(data)
    code_a = self.enc_a(data)
    return code_s, code_a

  def decode(self, code_s, code_a):
    return self.dec(code_s, code_a)

  def named_nets(self):
    return {
      "enc_s": self.enc_s,
      "enc_a": self.enc_a,
      "dec": self.dec,
      "dis_app": self.dis_app,
      "dis_pla": self.dis_pla
    }

  @staticmethod
  def create_place_encoder(args):
    raise NotImplementedError

  @staticmethod
  def create_appearance_encoder(args):
    raise NotImplementedError

  @staticmethod
  def create_decoder(args):
    raise NotImplementedError

  @staticmethod
  def create_place_domain_discriminator(args):
    raise NotImplementedError

  @staticmethod
  def create_appearance_compatibility_discriminator(args):
    raise NotImplementedError
