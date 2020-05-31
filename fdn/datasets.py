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

import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import torchvision


class MultiDomainDataset(torch.utils.data.Dataset):

  def __init__(self, datasets, shuffle=True):

    self.datasets = datasets
    self.shuffle = shuffle

    self.perms = []
    self.generate_permutation()

  def generate_permutation(self):

    def generate_permutation_once(n, target_len, shuffle):
      if self.shuffle:
        perm = torch.tensor([], dtype=torch.long)
        while perm.size(0) < self.__len__():
          rp = torch.randperm(n) if shuffle else torch.arange(0, n, dtype=torch.long)
          perm = torch.cat([perm, rp[0:min(n, target_len - perm.size(0))]])

      else:
        perm = torch.arange(0, self.__len__(), dtype=torch.long)
      return perm

    for dataset in self.datasets:
      self.perms.append(generate_permutation_once(len(dataset), self.__len__(), shuffle=self.shuffle))

  def __getitem__(self, index):
    datas_and_targets = [dataset[perm[index]] for dataset, perm in zip(self.datasets, self.perms)]
    datas, targets = zip(*datas_and_targets)
    return datas, targets

  def __len__(self):
    return max([len(ds) for ds in self.datasets])

  def num_domain(self):
    return len(self.datasets)

  @staticmethod
  def collate(batch):
    batch_list = []
    for data_and_target in batch:
      data = data_and_target[0]
      target = data_and_target[1]
      data1 = data[0]
      data2 = data[1]
      target1 = (-1, target[0])
      target2 = (-1, target[1])
      tmp = ((data1, data2), (target1, target2))
      batch_list.append(tmp)
    batch = default_collate(batch_list)
    return batch


class TwoDomainDataset(MultiDomainDataset):

  def __init__(self, dataset1, dataset2, shuffle=True):
    super().__init__([dataset1, dataset2], shuffle)


class ColoredMNIST(TwoDomainDataset):

  cmap = np.asarray([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
  ])

  mean = 0.1307
  std = 0.3081

  def num_domain(self):
    return ColoredMNIST.cmap.shape[0]

  @staticmethod
  def collate(batch):
    cmap_offset = (ColoredMNIST.cmap - 1) * ColoredMNIST.mean / ColoredMNIST.std

    cs = np.random.choice(len(ColoredMNIST.cmap), 2, replace=False)
    c1 = ColoredMNIST.cmap[cs[0]]
    c2 = ColoredMNIST.cmap[cs[1]]
    o1 = cmap_offset[cs[0]]
    o2 = cmap_offset[cs[1]]
    pair_list = []
    for batch_idx in range(len(batch)):
      data = batch[batch_idx][0]
      target = batch[batch_idx][1]
      data1 = data[0]
      data2 = data[1]
      target1 = target[0]
      target2 = target[1]
      data1 = torch.cat([data1 * c1[0] + o1[0], data1 * c1[1] + o1[1], data1 * c1[2] + o1[2]], dim=0)
      data2 = torch.cat([data2 * c2[0] + o2[0], data2 * c2[1] + o2[1], data2 * c2[2] + o2[2]], dim=0)
      target1 = (torch.tensor(cs[0]), target1)
      target2 = (torch.tensor(cs[1]), target2)
      pair = ((data1, data2), (target1, target2))
      pair_list.append(pair)
    batch = default_collate(pair_list)
    return batch


class PRDataset(MultiDomainDataset):

  def __init__(self, image_dirs, transform, shuffle=True):

    datasets = []
    for image_dir, domain in zip(image_dirs, range(len(image_dirs))):
      ds = SimpleImageFolder(image_dir, domain, transform)
      datasets.append(ds)

    super().__init__(datasets, shuffle)

  # def generate_permutation(self):
  #   def gen_perm(n, target_len, shuffle):
  #     perm = torch.tensor([], dtype=torch.long)
  #     while perm.size(0) < self.__len__():
  #       rp = torch.randperm(n) if shuffle else torch.arange(0, n, dtype=torch.long)
  #       perm = torch.cat([perm, rp[0:min(n, target_len - perm.size(0))]])
  #     return perm
  #
  #   for dataset in self.datasets:
  #     self.perms.append(gen_perm(len(dataset), self.__len__(), shuffle=self.shuffle))

  @staticmethod
  def collate(batch):
    ds_cnt = len(batch[0][0])
    #
    if ds_cnt == 2:
      cs = [0, 1]
    else:
      cs = np.random.choice(ds_cnt, 2, replace=False)

    batch_list = []
    for data_and_target in batch:
      data = data_and_target[0]
      target = data_and_target[1]
      data1 = data[cs[0]]
      data2 = data[cs[1]]
      target1 = target[cs[0]]
      target2 = target[cs[1]]
      tmp = ((data1, data2), (target1, target2))
      batch_list.append(tmp)
    batch = default_collate(batch_list)
    return batch


class GaussianND(torch.utils.data.Dataset):
  def __init__(self, s_mean, s_std, a_mean, a_std, projection_matrix, target, npoints=10000, shuffle=True):

    self.s_mean = torch.tensor(s_mean, dtype=torch.float)
    self.s_std = torch.tensor(s_std, dtype=torch.float)
    self.a_mean = torch.tensor(a_mean, dtype=torch.float)
    self.a_std = torch.tensor(a_std, dtype=torch.float)

    self.target = target
    self.npoints = npoints
    self.shuffle = shuffle

    self.projection_matrix = projection_matrix

  def _map(self, s, a):

    sa = torch.cat((s, a))
    sa = torch.squeeze(sa)
    mm = torch.matmul(self.projection_matrix, sa)
    mm = torch.unsqueeze(mm, 1)
    mm = torch.unsqueeze(mm, 2)
    return mm, sa

  def __getitem__(self, index):
    s = torch.randn_like(self.s_std) * self.s_std + self.s_mean
    s = s.unsqueeze(dim=1).unsqueeze(dim=2)
    a = torch.randn_like(self.a_std) * self.a_std + self.a_mean
    a = a.unsqueeze(dim=1).unsqueeze(dim=2)

    i, sa = self._map(s, a)

    return i, sa

  def __len__(self):
    return self.npoints


class SimpleImageFolder(torch.utils.data.Dataset):
  def __init__(self, image_dir, domain, transform=None, loader=torchvision.datasets.folder.default_loader):
    self.transform = transform
    self.loader = loader
    self.domain = domain

    self.samples = self.make_dataset(image_dir)
    if len(self.samples) == 0:
      raise (RuntimeError("Found 0 files in folder: " + image_dir + "\n"))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    path = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
      sample = self.transform(sample)

    target = (self.domain, path)

    return sample, target

  @staticmethod
  def make_dataset(root):
    images = []
    root = os.path.expanduser(root)

    for root, _, fnames in sorted(os.walk(root, followlinks=True)):
      for fname in sorted(fnames):
        if torchvision.datasets.folder.has_file_allowed_extension(fname, (".png", ".jpg", ".jpeg")):
          path = os.path.join(root, fname)
          item = path
          images.append(item)

    return images
