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

import toolz
import os
import re
import warnings
import collections
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from sklearn.metrics import precision_recall_curve, auc

import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd.profiler

from .loss import NSAdvLoss, LSAdvLoss, WAdvDivLoss, WAdvGPLoss, MSEWithHingeLoss
from .utils import PositionalDropout, batch_transform, create_network, check_match, smooth_pr
from .logging import LogWriter


class FDNTrainer(object):
  # noinspection PyTypeChecker
  def __init__(self, net_net, args):

    # Create model
    self.fdn = net_net.to(args.device)

    self.optimizers = {}
    if args.optimizer.type == "Adam":
      self.optimizers['ae'] = optim.Adam(list(self.fdn.enc_s.parameters()) +
                                         list(self.fdn.enc_a.parameters()) +
                                         list(self.fdn.dec.parameters()),
                                         lr=args.lr.ae, betas=(args.optimizer.beta1, args.optimizer.beta2),
                                         weight_decay=args.optimizer.weight_decay)
      self.optimizers['dis_app'] = optim.Adam(self.fdn.dis_app.parameters(),
                                              lr=args.lr.dis_app, betas=(args.optimizer.beta1, args.optimizer.beta2),
                                              weight_decay=args.optimizer.weight_decay)
      self.optimizers['dis_pla'] = optim.Adam(self.fdn.dis_pla.parameters(),
                                              lr=args.lr.dis_pla, betas=(args.optimizer.beta1, args.optimizer.beta2),
                                              weight_decay=args.optimizer.weight_decay)

    elif args.optimizer.type == "SGD":
      self.optimizers['ae'] = optim.SGD(list(self.fdn.enc_s.parameters()) +
                                        list(self.fdn.enc_a.parameters()) +
                                        list(self.fdn.dec.parameters()),
                                        lr=args.lr.ae, momentum=args.optimizer.momentum,
                                        weight_decay=args.optimizer.weight_decay)
      self.optimizers['dis_app'] = optim.SGD(self.fdn.dis_app.parameters(),
                                             lr=args.lr.dis_app, momentum=args.optimizer.momentum,
                                             weight_decay=args.optimizer.weight_decay)
      self.optimizers['dis_pla'] = optim.SGD(self.fdn.dis_pla.parameters(),
                                             lr=args.lr.dis_pla, momentum=args.optimizer.momentum,
                                             weight_decay=args.optimizer.weight_decay)

    else:
      raise NotImplementedError("Unknown optimizer type:%s" % args.optimizer.type)

    self.lr_schedulers = {}
    for (k, v) in self.optimizers.items():
      if args.lr.scheduler.type == "step":
        self.lr_schedulers[k] = optim.lr_scheduler.StepLR(v, args.lr.scheduler.step_size, args.lr.scheduler.gamma)
      elif args.lr.scheduler.type == "warmup":
        def get_warnup_lr(epoch):
          if epoch < args.lr.scheduler.warmup_epoch:
            return float(getattr(args.lr.scheduler.warmup_factor, k))
          else:
            return 1.0

        self.lr_schedulers[k] = optim.lr_scheduler.LambdaLR(v, lambda epoch: get_warnup_lr(epoch))
      elif args.lr.scheduler.type == "none":
        self.lr_schedulers[k] = None
      else:
        raise NotImplementedError("Unknown lr_scheduler name:%s" % args.lr.scheduler.type)

    # Criterion for reconstruction
    if args.recon_loss_type == "L2":
      self.recon_criterion = nn.MSELoss()
    elif args.recon_loss_type == "L1":
      self.recon_criterion = lambda x, y: torch.mean(torch.abs(x - y))
    elif args.recon_loss_type == "L2-hinge":
      self.recon_criterion = MSEWithHingeLoss()
    else:
      raise NotImplementedError("Unknown recon_loos type: %s" % args.recon_loss_type)

    # Criterion for adversarial
    if args.gan_type == "nsgan":
      self.dis_app_criterion = self.dis_pla_criterion = NSAdvLoss(args)
    elif args.gan_type == "lsgan":
      self.dis_app_criterion = self.dis_pla_criterion = LSAdvLoss(args)
    elif args.gan_type == "wgan-gp":
      self.dis_app_criterion = self.dis_pla_criterion = WAdvGPLoss(args)
    elif args.gan_type == "wgan-div":
      self.dis_app_criterion = self.dis_pla_criterion = WAdvDivLoss(args)
    else:
      raise NotImplementedError("Unknown adversarial loss type:%s" % args.gan_type)

    # DenoiseAE
    self.destructor = PositionalDropout(0.5) if args.denoise else None

    # Restore model
    self.epoch0 = 0
    self.step0 = 0
    self.global_step = 0
    if args.checkpoint != '':
      self.load(args.checkpoint, args.device)

    # Create log dir
    if args.evaluate_only:
      checkpoint_path = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.dirname(args.checkpoint)
      self.log_writer = LogWriter(full_dir=checkpoint_path + "/../eval/")
    else:
      self.log_writer = LogWriter(log_dir=args.log_dir, model_name=args.model_name)
      self.log_writer.backup_code(args)

    self.checkpoint_list = []

  def train(self, train_loader, eval_loader, args):

    if args.eval_mode == "epoch" and self.epoch0 == 0:
      self.evaluate_epoch(eval_loader, args)

    # Training loop
    for epoch in range(self.epoch0, args.epochs):

      # Update lr
      for _, v in self.lr_schedulers.items():
        if not (v is None):
          v.step(epoch)

      # Train one epoch
      with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
        self.train_epoch(train_loader, eval_loader, args, epoch)
      if prof is not None:
        print(prof.table(sort_by="self_cpu_time_total", row_limit=10))

      if args.eval_mode == "epoch" and (epoch % args.eval_interval == 0):
        self.evaluate_epoch(eval_loader, args)

      # Shuffle dataset
      train_loader.dataset.generate_permutation()
      eval_loader.dataset.generate_permutation()

  def evaluate(self, data_loader, args):

    if args.evaluate_only:

      checkpoint_path = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.dirname(args.checkpoint)
      self.fdn = create_network(args.dataset_type, args, checkpoint_path).to(args.device)

      total_params = sum(p.numel() for p in self.fdn.parameters())
      print("Total # of parameters:%d (%f M)" % (total_params, total_params / 1024 / 1024))

      if os.path.isfile(args.checkpoint):
        self.checkpoint_list.append(args.checkpoint)
      elif os.path.isdir(args.checkpoint):
        for f in os.listdir(args.checkpoint):
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
        print(f"Evaluating {chk_pt}")
        self.load(chk_pt, args.device)
        self.evaluate_epoch(data_loader, args)

    else:
      self.evaluate_epoch(data_loader, args)

  def train_epoch(self, train_loader, eval_loader, args, epoch):

    self.fdn.train()

    # Looop in 1 epoch
    for step, (data, target) in enumerate(train_loader):
      if epoch == self.epoch0:
        if step < self.step0:
          continue

      if args.eval_mode == 'step' and (self.global_step % args.eval_interval == 0):
        self.evaluate_epoch(eval_loader, args)

      data1 = data[0].to(args.device)
      data2 = data[1].to(args.device)

      with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
        self.train_batch(data1, data2, args, epoch, step)
      if prof is not None:
        print(prof.table(sort_by="self_cpu_time_total", row_limit=10))

      # Save model (step mode)
      if args.save_model_mode == 'step' and (self.global_step % args.save_model_interval == 0):
          model_filename = '%s/model-%d-%d.pth' % (self.log_writer.get_dir(), epoch, step)
          self.save(model_filename, epoch, step, self.global_step)

      self.global_step += 1

    # Save model (epoch mode)
    if args.save_model_mode == 'epoch' and (epoch % args.save_model_interval == 0):
        model_filename = '%s/model-%d-%d.pth' % (self.log_writer.get_dir(), epoch, step)
        self.save(model_filename, epoch, step, self.global_step)


  def evaluate_epoch(self, data_loader, args):

    print("This is %s on %s" % (self.log_writer.get_dir(), str(args.device)))

    self.fdn.eval()
    with torch.no_grad():
      self._evaluate_epoch_impl(data_loader, args)

  def _evaluate_epoch_impl(self, data_loader, args):

    if args.log.grad_histogram != 'none':
      for name, net in self.fdn.named_nets().items():
        param = toolz.first(net.parameters())
        self.log_writer.add_histogram(name + "/top", param.data.cpu().numpy(), self.global_step)
        if param.grad is not None:
          self.log_writer.add_histogram(name + "/top/grad", param.grad.data.cpu().numpy(), self.global_step)
        param = toolz.last(net.parameters())
        self.log_writer.add_histogram(name + "/bottom", param.data.cpu().numpy(), self.global_step)
        if param.grad is not None:
          self.log_writer.add_histogram(name + "/bottom/grad", param.grad.data.cpu().numpy(), self.global_step)

    if args.log.correlation != 'none' or args.log.pr != 'none' or args.log.viz_2d != 'none':
      code_s1_all = []
      code_s2_all = []
      code_a1_all = []
      code_a2_all = []
    else:
      code_s1_all = None
      code_s2_all = None
      code_a1_all = None
      code_a2_all = None

    if args.log.viz_2d != 'none':
      domain1_all = []
      domain2_all = []
    else:
      domain1_all = None
      domain2_all = None

    if args.log.viz_2d != 'none' or args.log.pr_match != 'none':
      label1_all = []
      label2_all = []
    else:

      label1_all = None
      label2_all = None

    if args.log.feature != 'none':
      data1_all = []
      data2_all = []
    else:
      data1_all = None
      data2_all = None

    if args.log.transform_image != 'none':
      transform_images = []
      ndom = data_loader.dataset.num_domain()
      for i in range(ndom):
        tmp = []
        for j in range(ndom):
          tmp.append(None)
        transform_images.append(tmp)
    else:
      transform_images = None

    for batch_idx, (data, target) in enumerate(data_loader):

      if (batch_idx + 1) > args.eval_batch:
        break

      print("\rEvaluating: %d/%d..." % (batch_idx + 1, len(data_loader)), end='')
      data1 = data[0].to(args.device)
      data2 = data[1].to(args.device)
      target1 = target[0]
      target2 = target[1]
      domain1 = target1[0]
      domain2 = target2[0]
      label1 = target1[1]
      label2 = target2[1]

      # Forward
      if code_s1_all is not None:
        code_s1, code_a1 = self.fdn.encode(data1)
        code_s2, code_a2 = self.fdn.encode(data2)
      else:
        code_s1 = code_a1 = code_s2 = code_a2 = None

      # Append output to list
      if code_s1_all is not None:
        code_s1_all.append(code_s1.cpu().numpy())
        code_s2_all.append(code_s2.cpu().numpy())
      if code_a1_all is not None:
        code_a1_all.append(code_a1.cpu().numpy())
        code_a2_all.append(code_a2.cpu().numpy())
      if domain1_all is not None:
        domain1_all.append(domain1.cpu().numpy())
        domain2_all.append(domain2.cpu().numpy())
      if label1_all is not None:
        if isinstance(label1, torch.Tensor):
          label1_all.append(label1.cpu().numpy())
          label2_all.append(label2.cpu().numpy())
        elif isinstance(label1, list):
          label1_all += label1
          label2_all += label2
        else:
          raise NotImplementedError
          # label1_all.append(label1)
          # label2_all.append(label2)
      if data1_all is not None:
        data1_all.append(data1.cpu().numpy())
        data2_all.append(data2.cpu().numpy())

      # For transform image
      if args.log.transform_image != 'none':

        dom1 = domain1[0].item()
        dom2 = domain2[0].item()

        if transform_images[dom1][dom2] is None or transform_images[dom2][dom1] is None:

          if transform_images[dom1][dom2] is None:
            data1_less = data1[:1]
            data2_less = data2[-1:]
          else:
            data1_less = data2[:1]
            data2_less = data1[-1:]
            dom1, dom2 = dom2, dom1

          code_s1, code_a1 = self.fdn.encode(data1_less)
          code_s2, code_a2 = self.fdn.encode(data2_less)
          code_a0 = torch.zeros_like(code_a1, device=code_a1.device)

          trans_img = self.fdn.decode(code_s1, code_a2)
          zero_app_img = self.fdn.decode(code_s1, code_a0)

          transform_images[dom1][dom2] = torch.cat(
            (torch.cat((data1_less, data2_less), dim=3), torch.cat((trans_img, zero_app_img), dim=3)), dim=2)
          transform_images[dom1][dom2] = batch_transform(transform_images[dom1][dom2], args.inv_transform)
          transform_images[dom1][dom2] = transform_images[dom1][dom2].squeeze(dim=0).permute([1, 2, 0]).cpu().numpy()

    print('done.')

    if args.log.correlation != 'none' or args.log.pr != 'none' or args.log.viz_2d != 'none':
      code_s1_all = np.concatenate(code_s1_all)
      code_s2_all = np.concatenate(code_s2_all)
    if args.log.correlation != 'none' or args.log.pr != 'none' or args.log.viz_2d != 'none':
      code_a1_all = np.concatenate(code_a1_all)
      code_a2_all = np.concatenate(code_a2_all)
    if args.log.viz_2d != 'none' or args.log.feature != 'none':
      label1_all = np.concatenate(label1_all)
      label2_all = np.concatenate(label2_all)
    if args.log.feature != 'none':
      domain1_all = np.concatenate(domain1_all)
      domain2_all = np.concatenate(domain2_all)
      data1_all = np.concatenate(data1_all)
      data2_all = np.concatenate(data2_all)

    def compute_pr(code1_all, code2_all):
      nc = code1_all.shape[0]
      code1_all_flatten = code1_all.reshape([code1_all.shape[0], -1]).copy()
      code2_all_flatten = code2_all.reshape([code2_all.shape[0], -1]).copy()
      code1_all_flatten /= np.linalg.norm(code1_all_flatten, axis=1, keepdims=True)
      code2_all_flatten /= np.linalg.norm(code2_all_flatten, axis=1, keepdims=True)
      _scores = np.matmul(code1_all_flatten, code2_all_flatten.transpose())

      _mscore = np.max(_scores, axis=0)
      _pick = np.argmax(_scores, axis=0)
      # Notice: Here we use args.place_threshold * 2 because check_match() will devide it by 2
      _correct = [1 if check_match(_pick[_i], _i, args.place_threshold * 2) else 0 for _i in range(0, nc)]

      _correctness = np.count_nonzero(_correct)

      _precision, _recall, threshold = precision_recall_curve(_correct, _mscore)
      _precision, _recall = smooth_pr(_precision, _recall)
      _curr_auc = auc(_recall, _precision)
      return _precision, _recall, _curr_auc, _scores, _mscore, _pick, _correctness, _correct

    if args.log.pr != 'none':
      precision, recall, curr_auc, scores, mscore, pick, correctness, correct = compute_pr(code_s1_all, code_s2_all)
    elif args.log.pr_match != 'none':
      _, _, _, _, _, pick, _, _ = compute_pr(code_s1_all, code_s2_all)

    if args.log.pr != 'none':
      n_all = pick.shape[0]
      accuracy = correctness / n_all

      if hasattr(args, "usetex") and args.usetex:
        matplotlib.rcParams['text.usetex'] = True

      plt.figure(1)
      plt.clf()
      plt.plot(recall, precision, '-.', label='AUC=%.2f' % curr_auc, linewidth=2)
      plt.title('PR')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.1])
      self.log_writer.add_figure("PR_smoothed", plt.gcf(), self.global_step, dest=args.log.pr)

      plt.figure(2)
      plt.clf()
      plt.imshow(scores, cmap='jet')
      plt.gca().xaxis.set_ticks_position('top')
      plt.colorbar()
      self.log_writer.add_figure("similarity_matrix", plt.gcf(), self.global_step, dest=args.log.pr)

      _, _, _, scores_a, _, pick_a, correctness_a, _ = compute_pr(code_a1_all, code_a2_all)
      plt.figure(3)
      plt.clf()
      plt.imshow(scores_a, cmap='jet')
      plt.gca().xaxis.set_ticks_position('top')
      plt.colorbar()
      self.log_writer.add_figure("similarity_matrix_of_a", plt.gcf(), self.global_step, dest=args.log.pr)

      self.log_writer.add_pr_curve("PR", correct, mscore, self.global_step)
      self.log_writer.add_scalar("AUC", curr_auc, self.global_step)
      self.log_writer.add_scalar("accuracy", accuracy, self.global_step)

      print("AUC=%.3f" % curr_auc)
      print("Accuracy=%d/%d=%.3f" % (correctness, n_all, accuracy))

    if args.log.pr_match != 'none':
      # self.log_writer.add_matrix("pick/%s" % args.model_name, pick.astype(int), self.global_step, dest=args.log.pr_match)
      N2 = len(data_loader.dataset.datasets[1])
      pick_m = pick[0:N2]
      mscore_m = mscore[0:N2]
      print(len(label1_all))
      label1_all_m = [label1_all[p] for p in pick_m]
      label2_all_m = label2_all[0:N2]
      label_all = ['%s %s %1.4f' % (label2, label1, score) for label1, label2, score in
                   zip(label1_all_m, label2_all_m, mscore_m)]
      self.log_writer.add_scalars("pick/%s" % args.model_name, dict(zip(label_all, pick.astype(int))), self.global_step,
                                  dest=args.log.pr_match)
      self.log_writer.add_scalars("correct/%s" % args.model_name, dict(zip(label_all, correct)), self.global_step,
                                  dest=args.log.pr_match)
      db_dir = self.log_writer.get_dir("pick_db/%s" % args.model_name)
      q_dir = self.log_writer.get_dir("pick_q/%s" % args.model_name)
      for i in range(N2):
        # print(label_all[i])
        label1, label2, _ = label_all[i].split(' ')
        # print(label1)
        # print(label2)
        shutil.copyfile(label1, "%s/%05d.png" % (q_dir, i))
        shutil.copyfile(label2, "%s/%05d.png" % (db_dir, i))

    if args.log.viz_2d != 'none':
      real_s1 = label1_all[:, 0:len(args.mean_s)]
      real_a1 = label1_all[:, -len(args.mean_a1):]
      real_s2 = label2_all[:, 0:len(args.mean_s)]
      real_a2 = label2_all[:, -len(args.mean_a2):]
      feat_s1 = code_s1_all.reshape([code_s1_all.shape[0], -1])
      feat_a1 = code_a1_all.reshape([code_a1_all.shape[0], -1])
      feat_s2 = code_s2_all.reshape([code_s2_all.shape[0], -1])
      feat_a2 = code_a2_all.reshape([code_a2_all.shape[0], -1])

      x1 = np.concatenate((real_s1, real_a1, feat_s1, feat_a1), axis=1)
      x2 = np.concatenate((real_s2, real_a2, feat_s2, feat_a2), axis=1)
      xylim = [-1.0 + min(np.min(x1), np.min(x2)), 1.0 + max(np.max(x1), np.max(x2))]
      xymax = max(abs(xylim[0]), abs(xylim[1]))
      x_label = []
      x_label += ['$\\hat{s}_%d$' % i for i in range(real_s1.shape[1])]
      x_label += ['$\\hat{a}_%d$' % i for i in range(real_a1.shape[1])]
      x_label += ['$s_%d$' % i for i in range(feat_s1.shape[1])]
      x_label += ['$a_%d$' % i for i in range(feat_a1.shape[1])]
      ndim = x1.shape[1]

      plt.figure(4, figsize=(10, 10))
      plt.clf()
      if hasattr(args, "usetex") and args.usetex:
        plt.rc('text', usetex=True)
      plt.rc('font', family='serif')
      for ind1 in range(ndim):
        for ind2 in range(0, ind1 + 1):
          plt.subplot(ndim, ndim, ind1 * ndim + ind2 + 1)
          plt.plot(x1[:, ind2], x1[:, ind1], 'r.', markersize=2)
          plt.plot(x2[:, ind2], x2[:, ind1], 'g.', markersize=2)

          plt.plot([-xymax, +xymax], [-xymax, +xymax], '--', color='black', linewidth=1.0, alpha=0.2)
          plt.plot([-xymax, +xymax], [+xymax, -xymax], '--', color='black', linewidth=1.0, alpha=0.2)

          plt.xlim(xylim)
          plt.ylim(xylim)

      for ind in range(ndim):
        plt.subplot(ndim, ndim, (ndim - 1) * ndim + ind + 1)
        plt.xlabel(x_label[ind])

        plt.subplot(ndim, ndim, ind * ndim + 1)
        plt.ylabel(x_label[ind])

      self.log_writer.add_figure("viz_2d", plt.gcf(), self.global_step, dest=args.log.viz_2d)

    if args.log.correlation != 'none':
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

      plt.figure(5, figsize=(3.45, 3.45), clear=True)
      if hasattr(args, "usetex") and args.usetex:
        plt.rc('text', usetex=True)
      plt.rc('font', family='Times', size=10)
      plt.tight_layout()
      cmap_name = 'bwr'
      plt.subplot(1, 2, 1)
      plt.imshow(cor1, vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
      plt.colorbar(orientation="horizontal", pad=0.2)
      plt.xlabel('$a_1$')
      plt.ylabel('$s$')
      plt.subplot(1, 2, 2)
      plt.imshow(cor2, vmin=-1, vmax=1, cmap=plt.get_cmap(cmap_name))
      plt.colorbar(orientation="horizontal", pad=0.2)
      plt.xlabel('$a_2$')
      plt.ylabel('$s$')
      plt.subplots_adjust(wspace=0.4)

      self.log_writer.add_figure('correlation', plt.gcf(), self.global_step, dest=args.log.correlation)

    if args.log.feature != 'none':
      matfile_data = {'data1': data1_all, 'data2': data2_all,
                      'domain1': domain1_all, 'domain2': domain2_all,
                      'digit1': label1_all, 'digit2': label2_all,
                      'code_s1': code_s1_all, 'code_s2': code_s2_all,
                      'code_a1': code_a1_all, 'code_a2': code_a2_all}
      hdf5storage.write(matfile_data, ".",
                        self.log_writer.get_dir(args.model_name) + "/features.mat", matlab_compatible=True)

    if args.log.transform_image != 'none':
      plt.figure(6)
      plt.clf()
      ndom = data_loader.dataset.num_domain()
      for i in range(ndom):
        for j in range(ndom):
          if i == j:
            continue
          if transform_images[i][j] is None:
            warnings.warn("transform_images[%d][%d] is None" % (i, j))
            continue
          plt.subplot(ndom, ndom, i * ndom + j + 1)
          plt.imshow(transform_images[i][j])
          plt.axis('off')
      self.log_writer.add_figure("transform_image", plt.gcf(), self.global_step, dest=args.log.transform_image)

  def train_batch(self, data1, data2, args, epoch, step):

    # Destruct
    if self.destructor is not None:
      data1 = self.destructor(data1).detach()
      data2 = self.destructor(data2).detach()

    if args.weight_dis_app != 0:
      loss_dis_app = self.update_dis_app(data1, data2, args)
    else:
      loss_dis_app = None

    if args.weight_dis_pla != 0:
      loss_dis_pla = self.update_dis_pla(data1, data2, args)
    else:
      loss_dis_pla = None

    if self.global_step % args.num_critic == 0:
      loss_recon = self.update_ae(data1, data2, args)
    else:
      loss_recon = None

    def digit_len(x):
      dllen = 1
      while x > 9:
        dllen += 1
        x /= 10
      return dllen

    tips = f'Epoch = {epoch:>0{digit_len(args.epochs)}}/{args.epochs} '
    tips += f'(step = {step:>0{digit_len(args.num_train_steps)}}/{args.num_train_steps}, '
    tips += f'global_step = {self.global_step:>0{digit_len(args.num_train_steps * args.epochs)}}) '
    if loss_dis_app is not None:
      tips += "loss_dis_app = %0.4f, " % loss_dis_app
    if loss_dis_pla is not None:
      tips += "loss_dis_pla = %0.4f, " % loss_dis_pla
    if loss_recon is not None:
      tips += "loss_recon = %0.4f" % loss_recon
    print("%s" % tips)

    # Add something to tensorboard
    if not (self.lr_schedulers['ae'] is None):
      self.log_writer.add_scalar("lr/AE", self.lr_schedulers['ae'].get_lr(), self.global_step)

  def update_dis_app(self, data1, data2, args):
    self.optimizers['dis_app'].zero_grad()

    code_s1, code_a1 = self.fdn.encode(data1)
    code_s2, code_a2 = self.fdn.encode(data2)
    dis_app_pred_true, code_f_true = self.fdn.dis_app(
      torch.cat((code_s1, code_s2), dim=0), torch.cat((code_a1, code_a2), dim=0))
    dis_app_pred_false, code_f_false = self.fdn.dis_app(
      torch.cat((code_s1, code_s2), dim=0), torch.cat((code_a2, code_a1), dim=0))

    if args.gan_type == "wgan-gp" or args.gan_type == "wgan-div":
      loss, wloss, gp = self.dis_app_criterion(
        dis_app_pred_true, dis_app_pred_false, self.fdn.dis_app, code_f_true, code_f_false)
    else:
      loss = self.dis_app_criterion(dis_app_pred_true, dis_app_pred_false)
      wloss = gp = torch.tensor(0.0)

    loss.backward()
    self.optimizers['dis_app'].step()

    loss = loss.item()
    wloss = wloss.item()
    gp = gp.item()

    self.log_writer.add_scalar("D_app/loss", loss, self.global_step)
    if args.gan_type == "wgan-gp" or args.gan_type == "wgan-div":
      self.log_writer.add_scalar("D_app/wloss", wloss, self.global_step)
      self.log_writer.add_scalar("D_app/gp", gp, self.global_step)

    return loss

  def update_dis_pla(self, data1, data2, args):
    self.optimizers['dis_pla'].zero_grad()

    code_s1, code_a1 = self.fdn.encode(data1)
    code_s2, code_a2 = self.fdn.encode(data2)
    code_s1_1, code_s1_2 = torch.split(code_s1, [args.batch_size // 2, args.batch_size // 2])
    code_s2_1, code_s2_2 = torch.split(code_s2, [args.batch_size // 2, args.batch_size // 2])
    dis_pla_pred_true, code_d_true = self.fdn.dis_pla(
      torch.cat((code_s1_1, code_s2_1), dim=0), torch.cat((code_s1_2, code_s2_2), dim=0))
    dis_pla_pred_false, code_d_false = self.fdn.dis_pla(
      torch.cat((code_s1_1, code_s2_1), dim=0), torch.cat((code_s2_1, code_s1_2), dim=0))

    if args.gan_type == "wgan-gp" or args.gan_type == "wgan-div":
      loss, wloss, gp = self.dis_pla_criterion(
        dis_pla_pred_true, dis_pla_pred_false, self.fdn.dis_pla, code_d_true, code_d_false)
    else:
      loss = self.dis_pla_criterion(dis_pla_pred_true, dis_pla_pred_false)
      wloss = gp = torch.tensor(0)

    loss.backward()
    self.optimizers['dis_pla'].step()

    loss = loss.item()
    wloss = wloss.item()
    gp = gp.item()

    self.log_writer.add_scalar("D_pla/loss", loss, self.global_step)
    if args.gan_type == "wgan-gp" or args.gan_type == "wgan-div":
      self.log_writer.add_scalar("D_pla/wloss", wloss, self.global_step)
      self.log_writer.add_scalar("D_pla/gp", gp, self.global_step)

    return loss

  def update_ae(self, data1, data2, args):

    self.optimizers['ae'].zero_grad()

    code_s1, code_a1 = self.fdn.encode(data1)
    code_s2, code_a2 = self.fdn.encode(data2)
    recon1 = self.fdn.decode(code_s1, code_a1)
    recon2 = self.fdn.decode(code_s2, code_a2)
    loss_recon = (self.recon_criterion(data1, recon1) + self.recon_criterion(data2, recon2)) / 2

    dis_app_pred_true, code_f_true = self.fdn.dis_app(
      torch.cat((code_s1, code_s2), dim=0), torch.cat((code_a1, code_a2), dim=0))
    dis_app_pred_false, code_f_false = self.fdn.dis_app(
      torch.cat((code_s1, code_s2), dim=0), torch.cat((code_a2, code_a1), dim=0))
    loss_dis_app = self.dis_app_criterion(dis_app_pred_false, dis_app_pred_true)

    code_s1_1, code_s1_2 = torch.split(code_s1, [args.batch_size // 2, args.batch_size // 2])
    code_s2_1, code_s2_2 = torch.split(code_s2, [args.batch_size // 2, args.batch_size // 2])
    dis_pla_pred_true, code_d_true = self.fdn.dis_pla(
      torch.cat((code_s1_1, code_s2_1), dim=0), torch.cat((code_s1_2, code_s2_2), dim=0))
    dis_pla_pred_false, code_d_false = self.fdn.dis_pla(
      torch.cat((code_s1_1, code_s2_1), dim=0), torch.cat((code_s2_1, code_s1_2), dim=0))
    loss_dis_pla = self.dis_pla_criterion(dis_pla_pred_false, dis_pla_pred_true)

    if args.weight_dis_app == 0.0 and args.weight_dis_pla == 0.0:
      loss = args.weight_recon * loss_recon
    else:
      loss = args.weight_dis_app * loss_dis_app + args.weight_dis_pla * loss_dis_pla + args.weight_recon * loss_recon

    loss.backward()
    self.optimizers['ae'].step()

    loss = loss.item()
    loss_dis_app = loss_dis_app.item()
    loss_dis_pla = loss_dis_pla.item()
    loss_recon = loss_recon.item()

    self.log_writer.add_scalar("AE/loss", loss, self.global_step)
    self.log_writer.add_scalar("AE/adv_app", loss_dis_app, self.global_step)
    self.log_writer.add_scalar("AE/adv_pla", loss_dis_pla, self.global_step)
    self.log_writer.add_scalar("AE/recon", loss_recon, self.global_step)

    return loss

  def load(self, checkpoint, device=None, from_old_format=False):

    def _remap_key(x: str):
      x = x.replace("fsn_", "fdn_")
      x = x.replace("gen_", "ae_")
      x = x.replace("fdis_", "dis_app_")
      x = x.replace("ddis_", "dis_pla_")
      x = x.replace("lenc.", "enc_s.")
      x = x.replace("genc.", "enc_a.")
      x = x.replace("gen.", "dec.")
      x = x.replace("fdis.", "dis_app.")
      x = x.replace("ddis.", "dis_pla.")
      return x

    def _remap_dict(x: collections.OrderedDict):
      x = {_remap_key(k): v for k, v in x.items()}
      return x

    def transfer_from_old(cp):
      cp = {_remap_key(k): (_remap_dict(v) if isinstance(v, collections.OrderedDict) else v) for k, v in cp.items()}
      return cp

    checkpoint = torch.load(checkpoint, device)
    if from_old_format:
      checkpoint = transfer_from_old(checkpoint)
    self.fdn.load_state_dict(checkpoint['fdn_state_dict'])
    self.optimizers['ae'].load_state_dict(checkpoint['ae_optimizer_state_dict'])
    self.optimizers['dis_app'].load_state_dict(checkpoint['dis_app_optimizer_state_dict'])
    self.optimizers['dis_pla'].load_state_dict(checkpoint['dis_pla_optimizer_state_dict'])
    self.epoch0 = checkpoint['epoch']
    self.step0 = checkpoint['step']
    self.global_step = checkpoint['global_step']

  def save(self, model_filename, epoch, step, global_step):
    torch.save({
      'fdn_state_dict': self.fdn.state_dict(),
      'ae_optimizer_state_dict': self.optimizers['ae'].state_dict(),
      'dis_app_optimizer_state_dict': self.optimizers['dis_app'].state_dict(),
      'dis_pla_optimizer_state_dict': self.optimizers['dis_pla'].state_dict(),
      'epoch': epoch,
      'step': step,
      'global_step': global_step
    }, model_filename)
