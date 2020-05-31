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

from os import listdir
import os
import re
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import hdf5storage
import matplotlib.pyplot as plt

import torch

from fdn.utils import create_network, check_match, smooth_pr, batch_transform


class FDNEvaluator(object):
  def __init__(self, net, args):

    self.checkpoint_list = []

    # If net is provided, use it in evaluation
    if net is not None:
      self.fdn = net

    # If net is not provided, evaluate network in saved checkpoint
    else:
      checkpoint_path = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.dirname(args.checkpoint)
      self.fdn = create_network(args.dataset_type, args, checkpoint_path)

      total_params = sum(p.numel() for p in self.fdn.parameters())
      print("Total # of parameters:%d (%f M)" % (total_params, total_params / 1024 / 1024))

      if os.path.isfile(args.checkpoint):
        self.checkpoint_list.append(args.checkpoint)
      elif os.path.isdir(args.checkpoint):
        for f in listdir(args.checkpoint):
          pattern = re.compile(".*\\.pth")
          if pattern.match(f):
            self.checkpoint_list.append(os.path.join(args.checkpoint, f))

      def get_key(x):
        _pattern = re.compile('model-(.*)-(.*)\\.pth')
        m = _pattern.findall(x)[0]
        return int(m[0]) * 1e10 + int(m[1])

      self.checkpoint_list.sort(key=get_key)
      print("%d checkpoint(s) found:" % len(self.checkpoint_list))
      for chk_pt in self.checkpoint_list:
        print(chk_pt)

  def evaluate(self, data_loader, log_writer, args, epoch=-1, step=-1, global_step=-1):

    self.fdn.eval()
    with torch.no_grad():

      if len(self.checkpoint_list) == 0:
        print("Start evaluation from current network ...")
        self.evaluate_model(data_loader, log_writer, "model-%d-%d" % (epoch, step), args)

      else:
        print("Start evaluation from checkpoint ...")
        for checkpoint_file in self.checkpoint_list:

          print("Loading checkpoint %s..." % checkpoint_file)
          checkpoint = torch.load(checkpoint_file)
          self.fdn.load_state_dict(checkpoint['net_state_dict'])

          model_name = re.compile('.*/train/(model-.*-.*).pth').findall(checkpoint_file)[0]

          self.evaluate_model(data_loader, log_writer.get_dir(model_name), model_name, args, global_step)

    self.fdn.train()

  def evaluate_model(self, data_loader, log_writer, model_name, args, global_step=-1):

    data1_all = []
    data2_all = []

    domain1_all = []
    domain2_all = []

    label1_all = []
    label2_all = []

    code_a1_all = []
    code_a2_all = []
    code_s1_all = []
    code_s2_all = []

    for batch_idx, (data, target) in enumerate(data_loader):

      if (batch_idx + 1) >= args.eval_batch:
        break

      print("%d/%d" % (batch_idx, len(data_loader)))
      data1 = data[0].to(args.device)
      data2 = data[1].to(args.device)
      target1 = target[0]
      target2 = target[1]
      domain1 = target1[0]
      domain2 = target2[0]
      label1 = target1[1]
      label2 = target2[1]

      # Forward
      code_s1 = self.fdn.enc_s(data1)
      code_s2 = self.fdn.enc_s(data2)
      code_a1 = self.fdn.enc_a(data1)
      code_a2 = self.fdn.enc_a(data2)
      code_a0 = torch.zeros_like(code_a1, device=code_a1.get_device())
      trans_1_2 = self.fdn.dec(code_s1, code_a2)
      trans_1_0 = self.fdn.dec(code_s1, code_a0)
      # recon1_1 = self.fdn.gen(code_s1, code_a1)
      # trans_2_1 = self.fdn.gen(code_s2, code_a1)
      # recon2_2 = self.fdn.gen(code_s2, code_a2)
      # trans_2_0 = self.fdn.gen(code_s2, code_a0)

      # inv_transform
      data1 = batch_transform(data1, args.inv_transform)
      data2 = batch_transform(data2, args.inv_transform)
      trans_1_2 = batch_transform(trans_1_2, args.inv_transform)
      trans_1_0 = batch_transform(trans_1_0, args.inv_transform)
      # recon1_1 = args.inv_transform(recon1_1)
      # trans_2_1 = args.inv_transform(trans_2_1)
      # recon2_2 = args.inv_transform(recon2_2)
      # trans_2_0 = args.inv_transform(trans_2_0)

      # Append output to list
      code_s1_all.append(code_s1.cpu().numpy())
      code_s2_all.append(code_s2.cpu().numpy())
      code_a1_all.append(code_a1.cpu().numpy())
      code_a2_all.append(code_a2.cpu().numpy())
      if args.plot_2d:
        domain1_all.append(domain1.cpu().numpy())
        domain2_all.append(domain2.cpu().numpy())
        label1_all.append(label1.cpu().numpy())
        label2_all.append(label2.cpu().numpy())
      if args.save_feature:
        data1_all.append(data1.cpu().numpy())
        data2_all.append(data2.cpu().numpy())

      if args.plot_image:

        for i in range(args.batch_size):
          dom1 = domain1[i].item()
          dom2 = domain2[i].item()

          data1_single = data1[i, :, :, :].cpu().numpy()
          data2_single = data2[i, :, :, :].cpu().numpy()
          trans_1_2_single = trans_1_2[i, :, :, :].cpu().numpy()
          trans_1_0_single = trans_1_0[i, :, :, :].cpu().numpy()
          image = torch.cat([torch.cat([data1_single, data2_single], dim=3),
                             torch.cat([trans_1_2_single,  trans_1_0_single], dim=3)], dim=2)
          image = image.squeeze(dim=0).permute(1, 2, 0)
          image = image.cpu().numpy()

          if args.plot_to_file:
            image_fn = '%s/%d_%d.png' % (log_writer.get_dir('image'), dom1 + 1, dom2 + 1)
            if not os.path.exists(image_fn):
              plt.imsave(image_fn, image)
          else:
            plt.imshow(image)

    # -------------------------------- Log something to tensorboard --------------------------------#
    if len(code_s1_all) == 0:
      code_s1_all = code_s2_all = code_a1_all = code_a2_all = \
        domain1_all = domain2_all = label1_all = label2_all = \
        data1_all = data2_all = np.empty([0, 0, 0, 0])
    else:
      code_s1_all = np.concatenate(code_s1_all)
      code_s2_all = np.concatenate(code_s2_all)
      code_a1_all = np.concatenate(code_a1_all)
      code_a2_all = np.concatenate(code_a2_all)
      if args.plot_2d:
        domain1_all = np.concatenate(domain1_all)
        domain2_all = np.concatenate(domain2_all)
        label1_all = np.concatenate(label1_all)
        label2_all = np.concatenate(label2_all)
      if args.save_feature:
        data1_all = np.concatenate(data1_all)
        data2_all = np.concatenate(data2_all)

    n_all = code_s1_all.shape[0]

    if args.log_pr or args.plot_pr:
      scores = np.matmul(
        code_s1_all.copy().reshape([code_s1_all.shape[0], -1]),
        code_s2_all.copy().reshape([code_s2_all.shape[0], -1]).transpose())

      mscore = np.max(scores, axis=0)
      pick = np.argmax(scores, axis=0)
      correct = [1 if check_match(pick[i], i, 6) else 0 for i in range(0, n_all)]

      correctness = np.count_nonzero(correct)
      print("Correctness/accuracy:%d/%f%%" % (correctness, correctness / n_all * 100))
      precision, recall, threshold = precision_recall_curve(correct, mscore)
      precision, recall = smooth_pr(precision, recall)
      curr_auc = auc(recall, precision)

      if args.log_pr:

        assert(global_step >= 0)

        def draw_pr(pre, rec):
          fig = plt.figure()
          plot = fig.add_subplot(111)
          cur_auc = auc(rec, pre)
          plot.plot(rec, pre, '-.', label='AUC=%.2f' % cur_auc, linewidth=2)
          plot.legend()
          plot.set_xlim([0, 1])
          plot.set_ylim([0, 1])
          fig.canvas.draw()

          buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
          buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
          buf = buf.transpose([2, 0, 1])
          return buf
        pr_fig = draw_pr(precision, recall)
        pr_fig = np.expand_dims(pr_fig, axis=0)
        log_writer.add_images("PR", pr_fig, global_step)
        log_writer.add_pr_curve("PR", correct, mscore, global_step)
        log_writer.add_scalar("AUC", curr_auc, global_step)

      if args.plot_pr:
        plt.figure(1)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(recall, precision, '-.', label='AUC=%.2f' % curr_auc, linewidth=2)
        plt.legend()
        plt.title(model_name)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.1])

        plt.subplot(2, 1, 2)
        plt.imshow(scores)
        plt.plot(range(0, n_all), pick, 'r+')
        plt.title('correctness=%d' % correctness)

        plt.plot(mscore, correct, '.')

        if args.plot_to_file:
          plt.savefig('%s/%s.png' % log_writer.get_dir(model_name), format="png")
        else:
          plt.show()

    # Plot something using PyPlot -------------------------------------------------------------#
    if args.plot_2d:
      # sa1_real = target1.cpu().numpy()
      # sa2_real = target2.cpu().numpy()
      # s1 = torch.squeeze(code_s1).cpu().numpy()
      # a1 = torch.squeeze(code_a1).cpu().numpy()
      # s2 = torch.squeeze(code_s2).cpu().numpy()
      # a2 = torch.squeeze(code_a2).cpu().numpy()
      real_s1 = label1_all[:, 0]
      real_a1 = label1_all[:, 1]
      real_s2 = label2_all[:, 0]
      real_a2 = label2_all[:, 1]

      plt.figure(2)
      plt.clf()
      dim_s = args.dim_enc_s
      dim_a = args.dim_enc_a
      h_plot = 1 + 2 * b * dim_s + dim_a
      w_plot = np.max(1, dim_s, dim_a)
      #
      plt.subplot(w_plot, h_plot, 1)
      plt.plot(real_s1, real_a1, 'x')
      plt.plot(real_s2, real_a2, 'o')
      plt.title('Real distribution')
      plt.xlim([-2, 2])
      plt.ylim([-2, 2])
      #
      for ind_a in range(dim_a):
        for ind_s in range(dim_s):
          plt.subplot(w_plot, h_plot, w_plot + ind_s * dim_a + ind_a)
          plt.plot(code_s1_all, code_a1_all, 'x')
          plt.plot(code_s2_all, code_a2_all, 'o')
          plt.xlim([-2, 2])
          plt.ylim([-2, 2])
          plt.title('s_%d vs s_%d' % (ind_s, ind_a))
      #
      plt.subplot(w_plot, h_plot, 1)
      plt.plot(s1, a1[:, 1], 'x')
      plt.plot(s2, a2[:, 1], 'o')
      plt.xlim([-2, 2])
      plt.ylim([-2, 2])
      plt.title('s vs a_1')
      #
      plt.subplot(w_plot, h_plot, 1)
      plt.plot(a1[:, 0], a1[:, 1], 'x')
      plt.plot(a2[:, 0], a2[:, 1], 'o')
      # plt.axis('equal')
      plt.xlim([-2, 2])
      plt.ylim([-2, 2])
      plt.title('a_0 vs a_1')

      # Option 1
      plt.draw()
      plt.pause(0.01)
      # Option 2
      pp = '%s/gaussian2d' % (args.log_dir_eval)
      if not os.path.isdir(pp):
        os.mkdir(pp)
      plt.savefig('%s/%05d.png' % (pp, int(self.global_step / args.eval_interval)), format="png")


    if args.plot_correlation:

      def peasonnr(x, y):
        x = x - np.mean(x, axis=0, keepdims=True)
        y = y - np.mean(y, axis=0, keepdims=True)
        x_norm = np.expand_dims(np.linalg.norm(x, axis=0), axis=1)
        y_norm = np.expand_dims(np.linalg.norm(y, axis=0), axis=0)
        return np.matmul(np.transpose(x), y) / x_norm / y_norm

      cor1 = peasonnr(code_s1_all.copy().reshape([code_s1_all.shape[0], -1]),
                      code_a1_all.copy().reshape([code_a1_all.shape[0], -1]))
      cor2 = peasonnr(code_s2_all.copy().reshape([code_s2_all.shape[0], -1]),
                      code_a2_all.copy().reshape([code_a2_all.shape[0], -1]))

      plt.figure(3)
      plt.clf()
      cmap_name = 'bwr'
      plt.subplot(1, 2, 1)
      plt.imshow(cor1, vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
      plt.colorbar()
      plt.title('COR1')
      plt.subplot(1, 2, 2)
      plt.imshow(cor2, vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
      plt.colorbar()
      plt.title('COR2')

      if args.plot_to_file:
        plt.savefig('%s/%s.png' % (log_writer.get_dir('correlation'), model_name), format="png")
      else:
        plt.draw()
        plt.pause(0.1)

    # -------------------------------- Save something --------------------------------#
    if args.save_feature:
      matfile_data = {'data1': data1_all, 'data2': data2_all,
                      'domain1': domain1_all, 'domain2': domain2_all,
                      'digit1': label1_all, 'digit2': label2_all,
                      'code_s1': code_s1_all, 'code_s2': code_s2_all,
                      'code_a1': code_a1_all, 'code_a2': code_a2_all}
      hdf5storage.write(matfile_data, ".", log_writer.get_dir(model_name) + "/features.mat", matlab_compatible=True)
