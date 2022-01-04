import os
import math
from decimal import Decimal
from glob import glob
import datetime, time
from importlib import import_module
import numpy as np

# import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import utils.utility as utility
from loss.contrast_loss import ContrastLoss
from loss.adversarial import Adversarial
from loss.perceptual import PerceptualLoss
from model.edsr import EDSR
from model.rcan import RCAN
from utils.niqe import niqe
from utils.ssim import calc_ssim
from data_aug import GaussianSmoothing, BatchGaussianNoise


class MultiCSDProTrainer:
    def __init__(self, args, loader, device):
        self.model_str = args.model.lower()
        self.pic_path = f'./output/{self.model_str}/{args.model_filename}/'
        if not os.path.exists(self.pic_path):
            self.makedirs = os.makedirs(self.pic_path)
        self.teacher_model = args.teacher_model
        self.checkpoint_dir = args.pre_train
        self.model_filename = args.model_filename
        self.model_filepath = f'{self.model_filename}.pth'
        self.writer = SummaryWriter(f'log/{self.model_filename}')

        self.start_epoch = -1
        self.device = device
        self.epochs = args.epochs
        self.init_lr = args.lr
        self.rgb_range = args.rgb_range
        self.scale = args.scale[0]
        self.stu_width_mult = args.stu_width_mult
        self.tea_width_mults = [1.0, 0.75, 0.5] #最后一个是contrastive里的positive
        self.stu_width_mults = [0.25]
        self.batch_size = args.batch_size
        self.neg_num = args.neg_num
        self.save_results = args.save_results
        self.self_ensemble = args.self_ensemble
        self.print_every = args.print_every
        self.best_psnr = 0
        self.best_psnr_epoch = -1

        self.loader = loader
        self.mean = [0.404, 0.436, 0.446]
        self.std = [0.288, 0.263, 0.275]

        self.build_model(args)
        self.upsampler = nn.Upsample(scale_factor=self.scale, mode='bicubic')
        self.optimizer = utility.make_optimizer(args, self.model)

        self.contra_lambda = args.contra_lambda
        self.ad_lambda = args.ad_lambda
        self.percep_lambda = args.percep_lambda
        self.kd_lambda = args.kd_lambda
        self.mean_outside = args.mean_outside
        self.t_detach = args.contrast_t_detach
        self.contra_loss = ContrastLoss(args.vgg_weight, args.d_func, self.t_detach)
        self.l1_loss = nn.L1Loss()

        self.gt_as_pos = args.gt_as_pos


    def train(self):
        self.model.train()

        total_iter = (self.start_epoch+1)*len(self.loader.loader_train)
        for epoch in range(self.start_epoch + 1, self.epochs):
            starttime = datetime.datetime.now()

            lrate = utility.adjust_learning_rate(self.optimizer, epoch, self.epochs, self.init_lr)
            print("[Epoch {}]\tlr:{}\t".format(epoch, lrate))
            psnr, t_psnr = 0.0, 0.0
            step = 0
            # stu_width_mult = self.stu_width_mults[0]
            stu_width_mult = self.stu_width_mult
            for batch, (lr, hr, _,) in enumerate(self.loader.loader_train):
                torch.cuda.empty_cache()
                step += 1
                total_iter += 1
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                self.optimizer.zero_grad()
                teacher_l1_loss = 0.0
                tea_psnr = {}
                tea_flops = {}
                positive = None
                flops = 0
                for width_mult in self.tea_width_mults:
                    tea_psnr[width_mult] = 0.0
                    tea_flops[width_mult] = 0.0
                for width_mult in self.tea_width_mults:
                    teacher_sr = self.model(lr, width_mult)
                    teacher_l1_loss += self.l1_loss(hr, teacher_sr)
                    tea_psnr[width_mult] += utility.calc_psnr(utility.quantize(teacher_sr, self.rgb_range), hr,  self.scale, self.rgb_range)
                    tea_flops[width_mult] += flops
                    positive = teacher_sr

                neg = self.upsampler(torch.flip(lr, [0]))
                neg = neg[:self.neg_num, :, :, :]
                l1_loss = 0.0
                contras_loss = 0.0
                # stu_psnr = 0.0
                stu_psnr = {}
                stu_flops = {}

                for width_mult in self.stu_width_mults:
                    stu_psnr[width_mult] = 0.0
                    stu_flops[width_mult] = 0.0
                for stu_width_mult in self.stu_width_mults:
                    student_sr = self.model(lr, stu_width_mult)

                    l1_loss += self.l1_loss(hr, student_sr)
                    contras_loss += self.contra_loss(positive, student_sr, neg)

                    student_sr = utility.quantize(student_sr, self.rgb_range)
                    stu_psnr[stu_width_mult] += utility.calc_psnr(student_sr, hr, self.scale, self.rgb_range)
                    stu_flops[stu_width_mult] += flops
                loss = l1_loss + self.contra_lambda * contras_loss + teacher_l1_loss

                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('Train/L1_loss', l1_loss, total_iter)
                self.writer.add_scalar('Train/Contras_loss', contras_loss, total_iter)
                self.writer.add_scalar('Train/Teacher_l1_loss', teacher_l1_loss, total_iter)
                self.writer.add_scalar('Train/Total_loss', loss, total_iter)

                if (batch + 1) % self.print_every == 0:
                    print(
                        f"[Epoch {epoch}/{self.epochs}] [Batch {batch * self.batch_size}/{len(self.loader.loader_train.dataset)}] "
                    )
                    for width_mult in self.tea_width_mults:
                        print(f"[t_psnr {width_mult}: {tea_psnr[width_mult] / step}], [flops {width_mult}: {tea_flops[width_mult] / step}]")
                    for width_mult in self.stu_width_mults:
                        print(f"[s_psnr {width_mult}: {stu_psnr[width_mult] / step}], [flops {width_mult}: {stu_flops[width_mult] / step}]")
                    utility.save_results(f'result_{batch}', hr, self.scale, width=1, rgb_range=self.rgb_range,
                                         postfix='hr', dir=self.pic_path)
                    utility.save_results(f'result_{batch}', student_sr, self.scale, width=1, rgb_range=self.rgb_range,
                                         postfix='s_sr', dir=self.pic_path)
                    
            print(f"training PSNR @epoch {epoch}: {stu_psnr[stu_width_mult] / step}")


            test_psnr = self.test(self.stu_width_mults[0])
            if test_psnr > self.best_psnr:
                print(f"saving models @epoch {epoch} with psnr: {test_psnr}")
                self.best_psnr = test_psnr
                self.best_psnr_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_psnr': self.best_psnr,
                    'best_psnr_epoch': self.best_psnr_epoch,
                }, f'{self.checkpoint_dir}{self.model_filepath}')

            endtime = datetime.datetime.now()
            cost = (endtime - starttime).seconds
            print(f"time of epoch{epoch}: {cost}")

    def test(self, width_mult=1):
        self.model.eval()
        with torch.no_grad():
            psnr = 0
            niqe_score = 0
            ssim = 0
            t0 = time.time()
            total_flops = 0
            flops = 0

            starttime = datetime.datetime.now()
            for d in self.loader.loader_test:
                # print(d[0], d[1], d[2])
                for lr, hr, filename in d:
                    lr = lr.to(self.device)
                    hr = hr.to(self.device)

                    x = [lr]
                    for tf in 'v', 'h', 't':
                        x.extend([utility.transform(_x, tf, self.device) for _x in x])
                    op = ['', 'v', 'h', 'hv', 't', 'tv', 'th', 'thv']

                    if self.self_ensemble:
                        res, flops, _ = self.model(lr, width_mult)
                        # res = self.model(lr, width_mult)
                        total_flops += flops
                        for i in range(1, len(x)):
                            _x = x[i]
                            _sr, flops, _ = self.model(_x, width_mult)
                            # _sr = self.model(_x, width_mult)
                            total_flops += flops
                            for _op in op[i]:
                                _sr = utility.transform(_sr, _op, self.device)
                            res = torch.cat((res, _sr), 0)
                        sr = torch.mean(res, 0).unsqueeze(0)
                    else:
                        sr, flops, _ = self.model(lr, width_mult)
                        # sr = self.model(lr, width_mult)
                        total_flops += flops

                    sr = utility.quantize(sr, self.rgb_range)
                    if self.save_results:
                        utility.save_results(str(filename), sr, self.scale, width_mult,
                                             self.rgb_range, 'SR',
                                             f'./output/test/{self.model_str}/{self.model_filename}')
                    psnr += utility.calc_psnr(sr, hr, self.scale, self.rgb_range, dataset=d)
                    print(filename, utility.calc_psnr(sr, hr, self.scale, self.rgb_range, dataset=d), calc_ssim(sr, hr, self.scale, dataset=d))
                    niqe_score += niqe(sr.squeeze(0).permute(1, 2, 0).cpu().numpy())
                    ssim += calc_ssim(sr, hr, self.scale, dataset=d)

                psnr /= len(d)
                niqe_score /= len(d)
                ssim /= len(d)
                total_flops /= len(d)
                print(width_mult, d.dataset.name, psnr, niqe_score, ssim, total_flops)

                endtime = datetime.datetime.now()
                cost = (endtime - starttime).seconds
                t1 = time.time()
                total_time = (t1 - t0)
                print(f"time of test: {total_time}")
                return psnr

    def build_model(self, args):
        m = import_module('model.' + self.model_str)
        self.model = getattr(m, self.model_str.upper())(args).to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=range(args.n_GPUs))
        self.load_model()

        # test teacher
        # self.test()

    def load_model(self):
        checkpoint_dir = self.checkpoint_dir
        print(f"[*] Load model from {checkpoint_dir}")
        if not os.path.exists(checkpoint_dir):
            self.makedirs = os.makedirs(checkpoint_dir)

        if not os.listdir(checkpoint_dir):
            print(f"[!] No checkpoint in {checkpoint_dir}")
            return

        model = glob(os.path.join(checkpoint_dir, self.model_filepath))

        no_student = False
        if not model:
            no_student = True
            print(f"[!] No checkpoint ")
            print("Loading pre-trained teacher model")
            model = glob(self.teacher_model)
            if not model:
                print(f"[!] No teacher model ")
                return

        model_state_dict = torch.load(model[0])
        if not no_student:
            self.start_epoch = model_state_dict['epoch']
            self.best_psnr = model_state_dict['best_psnr']
            self.best_psnr_epoch = model_state_dict['best_psnr_epoch']

        self.model.load_state_dict(model_state_dict['model_state_dict'], False)