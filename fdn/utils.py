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

import yaml
import argparse
import numpy as np
import random
import PIL.Image
# import cv2
from attrdict import AttrDict
import os
import importlib
import importlib.util


import torch.nn as nn
import torch
from torch.nn.functional import batch_norm


def config2arg(filename):
  parser = argparse.ArgumentParser(description='Default parse')

  with open(filename, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

  parser = _add_cfg_to_parser(cfg, parser)
  args = parser.parse_args()

  for arg_name in dir(args):
    if not(arg_name.startswith('__') or arg_name.startswith('_')):
      if arg_name.find('.') != -1:
        v = getattr(args, arg_name)
        _add_attr(args, arg_name, v)
        delattr(args, arg_name)

  return args


def _add_cfg_to_parser(cfg, parser, root=''):
  for k, v in cfg.items():
    rk = k if root == '' else root + '.' + k
    if isinstance(v, dict):
      parser = _add_cfg_to_parser(v, parser, root=rk)
    else:
      if isinstance(v, list):
        #TODO: only string list are support now
        parser.add_argument("--" + rk, type=str, default=v, action="append")
      else:
        parser.add_argument("--" + rk, type=type(v), default=v)

  return parser


def _add_attr(args, arg_name, value):
  pos = arg_name.find('.')
  if pos == -1:
    setattr(args, arg_name, value)
  else:
    arg_k = arg_name[:pos]
    arg_v = arg_name[(pos+1):]
    if hasattr(args, arg_k):
      attr = getattr(args, arg_k)
    else:
      attr = AttrDict()
      setattr(args, arg_k, attr)

    _add_attr(attr, arg_v, value)
    setattr(args, arg_k, attr)


def create_network(dataset_type, args, network_path=None):
  # Determine where to load networks
  if network_path is None:
    import fdn.networks as networks
  else:
    spec_file = os.path.join(network_path, 'fdn/networks/__init__.py')
    spec = importlib.util.spec_from_file_location('fdn.networks', spec_file)
    networks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(networks)

  network_name = dataset_type + "FDN"
  if hasattr(networks, network_name):
    network_class = getattr(networks, network_name)
    net = network_class(args)
    return net
  else:
    raise NotImplementedError("Cannot find network definition for %s" % dataset_type)


# Reference : https://github.com/rpng/calc/blob/master/TrainAndTest/writeDatabase.py
class RandomFourPointCrop(object):
  def __init__(self):
    pass

  def __call__(self, img):

    w = img.width
    h = img.height
    minsx = [0, 3 * w / 4]
    maxsx = [w / 4, w]
    minsy = [0, 3 * h / 4]
    maxsy = [h / 4, h]

    pts_orig = np.zeros((4, 2), dtype=np.float32)  # four original points
    pts_warp = np.zeros((4, 2), dtype=np.float32)  # points for the affine transformation.

    # fixed point for the first plane
    pts_orig[0, 0] = 0
    pts_orig[0, 1] = 0

    pts_orig[1, 0] = 0
    pts_orig[1, 1] = h

    pts_orig[2, 0] = w
    pts_orig[2, 1] = 0

    pts_orig[3, 0] = w
    pts_orig[3, 1] = h

    # random second plane
    pts_warp[0, 0] = random.uniform(minsx[0], maxsx[0])
    pts_warp[0, 1] = random.uniform(minsy[0], maxsy[0])

    pts_warp[1, 0] = random.uniform(minsx[0], maxsx[0])
    pts_warp[1, 1] = random.uniform(minsy[1], maxsy[1])

    pts_warp[2, 0] = random.uniform(minsx[1], maxsx[1])
    pts_warp[2, 1] = random.uniform(minsy[0], maxsy[0])

    pts_warp[3, 0] = random.uniform(minsx[1], maxsx[1])
    pts_warp[3, 1] = random.uniform(minsy[1], maxsy[1])

    # compute the 3x3 transform matrix based on the two planes of interest
    # trans = cv2.getPerspectiveTransform(pts_orig, pts_warp).flatten()[0:8]
    trans = getPerspectiveTransform(pts_orig, pts_warp).flatten()[0:8]

    # apply the perspective transormation to the image,
    # causing an automated change in viewpoint for the net's dual input
    img_warp = img.transform((w, h), PIL.Image.PERSPECTIVE, trans)

    return img_warp

  def __repr__(self):
    pass

def getPerspectiveTransform(pts1, pts2):

  A = np.zeros((8, 8))
  b = np.zeros((8, ))
  A[0:4, 0:2] = pts1
  A[0:4, 2] = 1
  A[0:4, 6:8] = - pts1 * pts2[:,0, np.newaxis]

  A[4:8, 3:5] = pts1
  A[4:8, 5] = 1
  A[4:8, 6:8] = - pts1 * pts2[:,1, np.newaxis]

  b[0:4] = pts2[:,0]
  b[4:8] = pts2[:,1]

  x = np.matmul(np.linalg.inv(A), b)

  T = np.eye(3, dtype=np.float32)
  T[0,:] = x[0:3]
  T[1,:] = x[3:6]
  T[2,0:2] = x[6:8]

  return T


class Clamp(object):
  def __init__(self, vmin, vmax):
    self.vmin = vmin
    self.vmax = vmax

  def __call__(self, tensor):
    return torch.clamp(tensor, self.vmin, self.vmax)

  def __repr__(self):
    return "Clamp(vmin=%f, vmax=%f)" % (self.vmin, self.vmax)


# Reference: https://github.com/mingyuliutw/UNIT/blob/25e99afe267df6eea2c97d23b05a42683d75e53c/networks.py
class Conv2dBlock(nn.Module):
  def __init__(self, input_dim, output_dim, kernel_size, stride,
               padding=0, normalizer='none', activation='relu', pad_type='zero'):
    super(Conv2dBlock, self).__init__()
    self.use_bias = True
    # initialize padding
    if pad_type == 'reflect':
      self.pad = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
      self.pad = nn.ReplicationPad2d(padding)
    elif pad_type == 'zero':
      self.pad = nn.ZeroPad2d(padding)
    else:
      assert 0, "Unsupported padding type: {}".format(pad_type)

    # initialize normalization
    norm_dim = output_dim
    if normalizer == 'bn':
      self.norm = nn.BatchNorm2d(norm_dim)
    elif normalizer == 'in':
      # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
      self.norm = nn.InstanceNorm2d(norm_dim)
    elif normalizer == 'ln':
      self.norm = LayerNorm(norm_dim)
    elif normalizer == 'adain':
      self.norm = AdaptiveInstanceNorm2d(norm_dim)
    elif normalizer == 'none':
      self.norm = None
    else:
      assert 0, "Unsupported normalization: {}".format(normalizer)

    # initialize activation
    if activation == 'relu':
      self.activation = nn.ReLU(inplace=False)
    elif activation == 'lrelu':
      self.activation = nn.LeakyReLU(0.2, inplace=False)
    elif activation == 'prelu':
      self.activation = nn.PReLU()
    elif activation == 'selu':
      self.activation = nn.SELU(inplace=True)
    elif activation == 'tanh':
      self.activation = nn.Tanh()
    elif activation == 'none':
      self.activation = None
    else:
      assert 0, "Unsupported activation: {}".format(activation)

    # initialize convolution
    self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

  def forward(self, x):
    x = self.conv(self.pad(x))
    if self.norm:
      x = self.norm(x)
    if self.activation:
      x = self.activation(x)
    return x


class PositionalDropout(nn.Module):
  def __init__(self, p=0.5):
    super().__init__()
    self.p = p

  def forward(self, x):
    mask = np.random.binomial(1, self.p, [x.size(0), 1, x.size(2), x.size(3)])
    y = x * torch.as_tensor(mask, device=x.device)
    return y


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1):
    super(AdaptiveInstanceNorm2d, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    # weight and bias are dynamically assigned
    self.weight = None
    self.bias = None
    # just dummy buffers, not used
    self.register_buffer('running_mean', torch.zeros(num_features))
    self.register_buffer('running_var', torch.ones(num_features))

  def forward(self, x):
    assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
    b, c = x.size(0), x.size(1)
    running_mean = self.running_mean.repeat(b)
    running_var = self.running_var.repeat(b)

    # Apply instance norm
    x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

    out = batch_norm(
      x_reshaped, running_mean, running_var, self.weight, self.bias,
      True, self.momentum, self.eps)

    return out.view(b, c, *x.size()[2:])

  def __repr__(self):
    return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
  def __init__(self, num_features, eps=1e-5, affine=True):
    super().__init__()
    self.num_features = num_features
    self.affine = affine
    self.eps = eps

    if self.affine:
      self.gamma = nn.Parameter(torch.as_tensor(num_features, dtype=torch.float).uniform_())
      self.beta = nn.Parameter(torch.zeros(num_features))

  def forward(self, x):
    shape = [-1] + [1] * (x.dim() - 1)
    if x.size(0) == 1:
      # These two lines run much faster in pytorch 0.4 than the two lines listed below.
      mean = x.view(-1).mean().view(*shape)
      std = x.view(-1).std().view(*shape)
    else:
      mean = x.view(x.size(0), -1).mean(1).view(*shape)
      std = x.view(x.size(0), -1).std(1).view(*shape)

    x = (x - mean) / (std + self.eps)

    if self.affine:
      shape = [1, -1] + [1] * (x.dim() - 2)
      x = x * self.gamma.view(*shape) + self.beta.view(*shape)
    return x


# Reference: https://github.com/rpng/calc/blob/2efc03636162bcd22047eadd7517c7dfd3344db5/TrainAndTest/testNet.py 
def smooth_pr(prec, rec):
  """
  Smooths precision recall curve according to TREC standards. Evaluates max precision at each 0.1 recall.
  Makes the curves look nice and not noisy
  """

  # n = len(prec)
  m = 11
  p_smooth = np.zeros((m,), dtype=np.float)
  r_smooth = np.linspace(0.0, 1.0, m)
  for i in range(m):
    j = np.argmin(np.absolute(r_smooth[i] - rec)) + 1
    p_smooth[i] = np.max(prec[:j])

  return p_smooth, r_smooth


def check_match(im_lab_k, db_lab, num_include):
  """
  Check if im_lab_k and db_lab are a match, i.e. the two images are less than or equal to
  num_include frames apart. The correct num_include to use depends on the speed of the camera,
  both for frame rate as well as physical moving speed.
  """
  if num_include == 1:
    if db_lab == im_lab_k:
      return True
  else:
    # This assumes that db_lab is a string of numerical characters, which it should be
    # print int(db_lab)-num_include/2, "<=", int(im_lab_k), "<=", int(db_lab)+num_include/2, "?"
    if (int(db_lab) - num_include / 2) <= int(im_lab_k) <= (int(db_lab) + num_include / 2):
      return True

  return False


def batch_transform(batch, transform):
  for i in range(batch.shape[0]):
    batch[i] = transform(batch[i])
  return batch
