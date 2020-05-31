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

import time
import os
import shutil
import yaml
import sys
import pathlib
import logging
import numpy as np

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch


class LogWriter(SummaryWriter):

  def __init__(self, log_dir="", model_name="", train=True, full_dir=None):

    if full_dir is None:
      current_time = time.time()
      log_dir = "%s/%s/%f/%s" % (log_dir, model_name, current_time, 'train' if train else 'eval')
    else:
      log_dir = full_dir

    super().__init__(logdir=log_dir)

  def add_figure(self, tag, figure, global_step=None, close=True, walltime=None, dest='tb'):
    if dest == 'tb':
      super().add_figure(tag=tag, figure=figure, global_step=global_step, close=close, walltime=walltime)
    elif dest == 'pyplot':
      figure.show()
      plt.pause(0.1)
    elif dest == 'file':
      figure.savefig('%s/%08d.eps' % (self.get_dir(tag), global_step), format="eps", bbox_inches='tight', pad_inches=0)
    else:
      raise NotImplementedError

  def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW', dest="tb"):
    if dest == 'tb':
      super().add_images(tag=tag, img_tensor=img_tensor, global_step=global_step, walltime=walltime,
                         dataformats=dataformats)
    elif dest == 'pyplot':
      # TODO: now plotting only the first image
      image = img_tensor[0].permute(1, 2, 0).cpu().numpy()
      plt.imshow(image)
      plt.show()
      plt.pause(0.1)
    elif dest == 'file':
      # TODO: now saving only the first image
      image = img_tensor[0].permute(1, 2, 0).cpu().numpy()
      plt.imsave('%s/%08d.png' % (self.get_dir(tag), global_step), image)
    else:
      raise NotImplementedError

  def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW', dest='tb'):
    if dest == 'tb':
      super().add_image(tag=tag, img_tensor=img_tensor, global_step=global_step, walltime=walltime,
                        dataformats=dataformats)
    elif dest == 'pyplot':
      image = img_tensor.permute(1, 2, 0).cpu().numpy()
      plt.imshow(image)
      plt.show()
      plt.pause(0.1)
    elif dest == 'file':
      image = img_tensor.permute(1, 2, 0).cpu().numpy()
      plt.imsave('%s/%08d.png' % (self.get_dir(tag), global_step), image)
    else:
      raise NotImplementedError

  def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None, dest='tb'):
    print('Here:%s' % dest)
    if dest == 'tb':
      super().add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
    elif dest == 'file':
      # if isinstance(tag_scalar_dict, torch.Tensor):
        # matrix = matrix.cpu().numpy()
      # np.savetxt('%s/%08d.txt' % (self.get_dir(tag), global_step), matrix, fmt="%d")
      # print(tag_scalar_dict)
      with open('%s/%08d.txt' % (self.get_dir(main_tag), global_step), 'w') as f:
        for t, s in tag_scalar_dict.items():
          f.write('%s %d\n' % (t, s))
    else:
      raise NotImplementedError

  def backup_code(self, args, src_exts=("py", "yaml")):
    # Save arguments
    with open(os.path.join(self.get_dir(), "args.yaml"), "w") as f:
      yaml.dump(args, f)
    # Save source files
    main_dir = pathlib.Path(os.path.dirname(os.path.realpath(sys.modules['__main__'].__file__)))
    for ext in src_exts:
      for src in main_dir.glob(f'**/*.{ext}'):
        if src.is_file():
          dest = pathlib.Path(self.get_dir(), src.relative_to(main_dir))
          if not dest.parent.exists():
            logging.info(f"Creating {dest.parent}")
            dest.parent.mkdir(mode=0o755, parents=True)
          logging.info(f"Copying {src} to {dest}")
          shutil.copy2(str(src), str(dest))

  def get_dir(self, *args):
    if len(args) == 0:
      return self.logdir
    else:
      full_dir = os.path.join(self.logdir, *args)
      if not os.path.isdir(full_dir):
        os.makedirs(full_dir, 0o755)
      return full_dir
