import torch
from torch.utils.data import DataLoader
import os

import utils.utility as utility
import data
from data.neg_sample import Neg_Dataset
import loss
from option import args
from trainer.slim_contrast_trainer import SlimContrastiveTrainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)

if __name__ == '__main__':
    loader = data.Data(args)
    neg_loader = DataLoader(dataset=Neg_Dataset(os.path.join(args.dir_data, f'DIV2K/DIV2K_train_LR_bicubic/X{args.scale[0]}')), batch_size=10,
                            shuffle=True,
                            pin_memory=True, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.seperate:
        t = SeperateContrastiveTrainer(args, loader, device)
    else:
        # t = SlimContrastiveTrainer(args, loader, device)
        t = SlimContrastiveTrainer(args, loader, device, neg_loader)

    if args.model_stat:
        total_param = 0
        if args.seperate:
            for name, param in t.s_model.named_parameters():
                total_param += torch.numel(param)
        else:
            for name, param in t.model.named_parameters():
                # print(name, '      ', torch.numel(param))
                total_param += torch.numel(param)
        print(total_param)



    if not args.test_only:
        t.train()
    else:
        t.test(args.stu_width_mult)
    checkpoint.done()