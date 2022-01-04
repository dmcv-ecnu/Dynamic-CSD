import os
from glob import glob
import datetime
import time

from option import args
from model.edsr import EDSR
from model.carn import CARN
from model.rcan import RCAN
import data
from utils import utility

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate(sr_model, test_loader, width_list, args, geneinfo, device, self_ensemble=False, save_results=False):
    scale = args.scale[0]
    sr_model.eval()
    upsampler = nn.Upsample(scale_factor=scale, mode='bicubic')
    width_cnt = {}
    for width in width_list:
        width_cnt[width] = 0
    with torch.no_grad():
        psnr = 0
        niqe_score = 0
        ssim = 0
        total_flops = 0

        for d in test_loader:
            for lr, hr, filename in d:
                lr = lr.to(device)
                hr = hr.to(device)

                llr = F.interpolate(lr, scale_factor=1.0 / scale, mode='bicubic').clamp(min=0, max=args.rgb_range)
                lbic = upsampler(llr)
                lbic = utility.quantize(lbic, args.rgb_range)
                crop_w = lr.shape[-2] - lbic.shape[-2]
                crop_h = lr.shape[-1] - lbic.shape[-1]
                lhr = lr
                if crop_w > 0:
                    lhr = lhr[:, :, int(crop_w / 2):-(crop_w - int(crop_w / 2)), :]
                if crop_h > 0:
                    lhr = lhr[:, :, :, int(crop_h / 2):-(crop_h - int(crop_h / 2))]
                bic_psnr = utility.calc_psnr(lbic, lhr, scale, args.rgb_range, dataset=d)
                # print(bic_psnr)

                width_mult = width_list[-1]
                for i, gene in enumerate(geneinfo):
                    if bic_psnr > gene:
                        width_mult = width_list[i]
                        break
                width_cnt[width_mult] += 1
                print(filename, bic_psnr, width_mult)

                x = [lr]
                for tf in 'v', 'h', 't':
                    x.extend([utility.transform(_x, tf, device) for _x in x])
                op = ['', 'v', 'h', 'hv', 't', 'tv', 'th', 'thv']

                if self_ensemble:
                    res, flops, _ = sr_model(lr, width_mult)
                    total_flops += flops
                    # res, _ = self.model(lr, width_mult)
                    for i in range(1, len(x)):
                        _x = x[i]
                        # _sr, _ = self.model(_x, width_mult)
                        _sr, flops, _ = sr_model(_x, width_mult)
                        total_flops += flops
                        for _op in op[i]:
                            _sr = utility.transform(_sr, _op, device)
                        res = torch.cat((res, _sr), 0)
                    sr = torch.mean(res, 0).unsqueeze(0)
                else:
                    sr, flops, _ = sr_model(lr, width_mult)
                    total_flops += flops
                    # sr, _ = self.model(lr, width_mult)

                sr = utility.quantize(sr, args.rgb_range)
                if save_results:
                    #     if not os.path.exists(f'./output/test/{self.model_str}/{self.model_filename}'):
                    #         self.makedirs = os.makedirs(f'./output/test/{self.model_str}/{self.model_filename}')
                    utility.save_results(str(filename), sr, scale, width_mult,
                                         args.rgb_range, 'SR')
                    utility.save_results(str(filename), lbic, scale, width_mult,
                                         args.rgb_range, 'LBic')

                # print(
                #     f"{filename}\t {width_mult}\t {utility.calc_psnr(sr, hr, self.scale, self.rgb_range, dataset=d)}\t ")
                psnr += utility.calc_psnr(sr, hr, scale, args.rgb_range, dataset=d)

            psnr /= len(d)
            niqe_score /= len(d)
            ssim /= len(d)
            # print(width_mult, d.dataset.name, psnr, niqe_score, ssim)
            # print(f"Total FLOPs: {total_flops}")

            print(width_cnt)
            return psnr, total_flops/len(d)

def GA_test():
    width_list = [0.25, 0.5, 0.75, 1.0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # sr_model = CARN(args).to(device)
    # sr_model = EDSR(args).to(device)
    sr_model = RCAN(args).to(device)
    sr_model = nn.DataParallel(sr_model, device_ids=range(args.n_GPUs))
    checkpoint_dir = args.pre_train
    print(f"[*] Load model from {checkpoint_dir}")
    if not os.listdir(checkpoint_dir):
        print(f"[!] No checkpoint in {checkpoint_dir}")
        return
    model = glob(os.path.join(checkpoint_dir, f'{args.model_filename}.pth'))

    model_state_dict = torch.load(model[0])
    sr_model.load_state_dict(model_state_dict['model_state_dict'], False)
    loader = data.Data(args)

    threshold = args.threshold
    starttime = time.time()
    psnr, flops = evaluate(sr_model, loader.loader_test, width_list, args,
                           threshold, device)
    endtime = time.time()
    cost = (endtime - starttime)
    print(f"Time cost:{cost}")
    print(f"Evaluate on {args.data_test}: psnr:{psnr}, flops:{flops}")

if __name__ == '__main__':
    GA_test()
