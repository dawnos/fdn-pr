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

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy, sigmoid
import torch.autograd as autograd


class AdversarialLoss(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.device = args.device

  def forward(self, *args):
    raise NotImplementedError


class BasicAdvLoss(AdversarialLoss):
  def __init__(self, args):
    super().__init__(args)
    self.smooth_label = args.smooth_label

  def forward(self, pred_true, pred_false):
    if self.smooth_label:
      target_true = None if pred_true is None else _rand_like_interval(pred_true, 0.8, 1.0).to(self.device)
      target_false = None if pred_false is None else torch.zeros_like(pred_false).to(self.device)
    else:
      target_true = None if pred_true is None else torch.ones_like(pred_true).to(self.device)
      target_false = None if pred_false is None else torch.zeros_like(pred_false).to(self.device)

    if pred_true is None:
      return self._loss_impl(pred_false, target_false)
    elif pred_false is None:
      return self._loss_impl(pred_true, target_true)
    else:
      return (self._loss_impl(pred_true, target_true) + self._loss_impl(pred_false, target_false)) / 2

  def _loss_impl(self, pred, target):
    raise NotImplementedError


class NSAdvLoss(BasicAdvLoss):
  def __init__(self, args):
    super().__init__(args)
    self.criterion = nn.CrossEntropyLoss().to(args.device)

  def _loss_impl(self, pred, target):
    return torch.mean(binary_cross_entropy(sigmoid(pred), target))


class LSAdvLoss(BasicAdvLoss):
  def __init__(self, args):
    super().__init__(args)

  def _loss_impl(self, pred, target):
    return torch.mean((pred - target) ** 2)


class WAdvGPLoss(AdversarialLoss):
  def __init__(self, args):
    super().__init__(args)
    self.weight = args.weight_gp

  def forward(self, pred_true, pred_false, dis_net=None, code_true=None, code_false=None):
    wloss = 0
    if not (pred_true is None):
      wloss += - torch.mean(pred_true)
    if not (pred_false is None):
      wloss += torch.mean(pred_false)

    if not (code_true is None or code_false is None):
      gp = self.get_gradient_penalty(dis_net, code_true, code_false)
      loss = wloss + self.weight * gp
      # loss += self.weight * self.get_gradient_penalty(dis_net, code_true, code_false)
      return loss, wloss, gp
    else:
      loss = wloss
      return loss

  def get_gradient_penalty(self, dis_net, data_true, data_false):
    alpha = torch.rand((data_true.shape[0], 1, 1, 1)).to(self.device)
    interpolates = alpha * data_true + (torch.ones_like(alpha) - alpha) * data_false
    pred_interpolates = dis_net.predict(interpolates)
    gradients = autograd.grad(pred_interpolates, interpolates,
                              grad_outputs=torch.ones(pred_interpolates.size()).to(self.device),
                              retain_graph=True, create_graph=True, only_inputs=True)[0]
    slopes = torch.norm(gradients, dim=1)

    gradient_penalty = torch.mean((slopes - 1) ** 2)

    return gradient_penalty


class WAdvDivLoss(WAdvGPLoss):
  def __init__(self, args):
    super().__init__(args)
    self.weight = args.weight_div

  def get_gradient_penalty(self, dis_net, data_true, data_false):

    def _get_slopes(net, data):
      pred = net.predict(data)
      gradients = autograd.grad(pred, data,
                                grad_outputs=torch.ones(pred.size()).to(self.device),
                                retain_graph=True, create_graph=True, only_inputs=True)[0]
      slopes = torch.norm(gradients, dim=1)
      return slopes

    slopes_true = _get_slopes(dis_net, data_true)
    slopes_false = _get_slopes(dis_net, data_false)

    gradient_penalty = (torch.mean((slopes_true - 0) ** 2) + torch.mean((slopes_false - 0) ** 2)) / 2

    return gradient_penalty


def _rand_like_interval(x, vmin, vmax):
  return torch.rand_like(x) * (vmax - vmin) + vmin


class MSEWithHingeLoss(nn.Module):

  def __init__(self):
    super().__init__()

    self.mse = nn.MSELoss()
    self.relu = nn.ReLU()

  def forward(self, input: torch.Tensor, target: torch.Tensor):
    loss1 = self.mse(input, target)
    loss2 = self.relu(input - torch.ones_like(input)) ** 2
    loss3 = self.relu(-input - torch.ones_like(input)) ** 2
    loss = loss1 + 10.0 * (loss2 + loss3)
    return loss
