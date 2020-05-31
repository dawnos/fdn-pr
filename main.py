#!/usr/bin/env python

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

import sys
import numpy as np

import torch
import torch.cuda
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.utils
import torch.utils.data
import torch.backends.cudnn

from fdn.trainer import FDNTrainer
from fdn.datasets import PRDataset, TwoDomainDataset, ColoredMNIST, GaussianND
from fdn.utils import Clamp, RandomFourPointCrop
from fdn.utils import config2arg, create_network

import matplotlib
matplotlib.use('Agg')


def main():
  if len(sys.argv) < 2:
    raise Exception("Too few argument. Usage: python %s config-file-name" % sys.argv[0])
  conf_file = sys.argv.pop(1)
  args = config2arg(conf_file)

  # Determine device
  if torch.cuda.is_available() and args.gpu != "cpu":
    if args.gpu == "auto":
      print("Trying to find valid GPU automatically...")
      for dev_id in range(torch.cuda.device_count()):
        try:
          args.device = torch.device("cuda:%d" % dev_id)
          with torch.cuda.device(dev_id):
            _cache = torch.tensor((), dtype=torch.uint8, device=args.device)
            _cache.new_empty((int(torch.cuda.get_device_properties(args.device).total_memory * 0.9), 1))
            del _cache
            torch.cuda.empty_cache()
            break
        except RuntimeError:
          args.device = None
          print("Device cuda:%d not avaliable" % dev_id)
      if args.device is None:
        print("No valid GPU found")
        exit(-1)
      else:
        print("Find valid GPU: cuda:%d" % dev_id)
    else:
      print("Using given GPU: %s" % args.gpu)
      args.device = torch.device(args.gpu)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark = True
  else:
    print("Using CPU")
    args.device = torch.device("cpu")

  if hasattr(args, 'seed'):
    torch.manual_seed(args.seed)

  # Setup dataset & dataloader
  if args.dataset_type == "GaussianND":
    dim_x = args.dim_x
    s_mean = args.s.mean
    s_std = args.s.std
    a1_mean = args.a1.mean
    a1_std = args.a1.std
    a2_mean = args.a2.mean
    a2_std = args.a2.std

    projection_matrix = torch.randn((dim_x, len(s_mean) + len(a1_mean)))

    train_dataset = TwoDomainDataset(
      GaussianND(s_mean, s_std, a1_mean, a1_std, projection_matrix, 0),
      GaussianND(s_mean, s_std, a2_mean, a2_std, projection_matrix, 1))

    eval_dataset = TwoDomainDataset(
      GaussianND(s_mean, s_std, a1_mean, a1_std, projection_matrix, 0),
      GaussianND(s_mean, s_std, a2_mean, a2_std, projection_matrix, 1))

    args.inv_transform = lambda x: x

  elif args.dataset_type == "MNIST":

    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((ColoredMNIST.mean,), (ColoredMNIST.std,))
    ])

    train_dataset = ColoredMNIST(
      MNIST(args.train_data_dir[0], train=True, download=True, transform=transform),
      MNIST(args.train_data_dir[1], train=True, download=True, transform=transform))
    eval_dataset = ColoredMNIST(
      MNIST(args.eval_data_dir[0], train=False, transform=transform),
      MNIST(args.eval_data_dir[1], train=False, transform=transform))

    args.inv_transform = transforms.Compose([
      transforms.Normalize((-ColoredMNIST.mean / ColoredMNIST.std,), (1.0 / ColoredMNIST.std,)),
      Clamp(0.0, 1.0),
    ])

  elif args.dataset_type == "PlaceRecognition":
    mean_rgb = np.asarray([0.5, 0.5, 0.5])
    std_rgb = np.asarray([0.5, 0.5, 0.5])
    transform1_list = []
    if args.augmentation:
      transform1_list += [
        transforms.RandomHorizontalFlip(),
        RandomFourPointCrop(),
      ]
    transform1_list += [
      transforms.Resize([128, 128]),
      transforms.ToTensor(),
      transforms.Normalize(mean_rgb, std_rgb),
    ]
    transform1 = transforms.Compose(transform1_list)
    transform2 = transforms.Compose([
      transforms.Resize([128, 128]),
      transforms.ToTensor(),
      transforms.Normalize(mean_rgb, std_rgb),
    ])

    train_dataset = PRDataset(args.train_data_dir, transform=transform1, shuffle=True)
    eval_dataset = PRDataset(args.eval_data_dir, transform=transform2, shuffle=False)

    args.inv_transform = transforms.Compose([
      transforms.Normalize(-mean_rgb / std_rgb, 1 / std_rgb),
      Clamp(0.0, 1.0)
    ])

  else:
    raise ValueError("Unknown dataset:%s" % args.dataset_type)

  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, collate_fn=train_dataset.collate,
                                             batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                             pin_memory=True, drop_last=True)
  eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, collate_fn=eval_dataset.collate,
                                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                            pin_memory=True, drop_last=False)

  args.num_train_steps = len(train_loader)
  args.num_eval_steps = len(eval_loader)

  net = create_network(args.dataset_type, args)

  trainer = FDNTrainer(net, args)
  if not args.evaluate_only:
    trainer.train(train_loader, eval_loader, args)
  else:
    trainer.evaluate(eval_loader, args)


if __name__ == "__main__":
  main()
