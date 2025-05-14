from tqdm import tqdm
import numpy as np
import os
import glob
import time

import torch
from torch import nn
from torch.optim import Adam

from train.train_base import BaseTrainer
from models.swinjscc import *
#from modules.distortion import Distortion

import torch
from torch.optim import Adam
from tqdm import tqdm

#from models.swinjscc import SWINJSCC
from models.eDjscc import *
#from losses import Distortion  # hoặc nơi bạn định nghĩa loss của SwinJSCC

class BF_CNNTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        # 1) Deep JSCC generator -- frozen
        self.net = Generator(args).to(self.device)
        # 2) Bias-free denoiser
        self.model = BF_CNN(1, 64, 3, 20, args.num_channels).to(self.device)
        # 3) Optimizer chỉ cho denoiser
        self.opt_den  = torch.optim.Adam(self.model.parameters(), lr=args.lr,betas=(0.9, 0.999), weight_decay=0.0)
        self.criterion = nn.MSELoss(reduction="mean")
        # noise power
        self.base_snr = args.base_snr
        self.stddev = 10 ** (-0.05 * args.base_snr)

    def train(self):
        print("base snr:", self.base_snr)
        print("stddev:", self.stddev)
        print_freq = 100  # Set print frequency
        for epoch in range(self.args.out_e):
            self.net.eval()       # freeze JSCC
            self.model.train()    # train denoiser
            running_loss = 0.0

            for i, (x, _) in enumerate(tqdm(self.train_dl, desc=f"Denoiser Ep{epoch}")):
                x = x.to(self.device)
                with torch.no_grad():
                    code = self.net.encoder(x)                   # latent
                    noise = torch.randn_like(code) * self.stddev # sampled noise
                residual = self.model(code + noise)              # predict residual

                loss_codeword = self.criterion(residual, noise)  # MSE(res, noise)
                loss = loss_codeword
                self.opt_den.zero_grad()
                loss.backward()
                self.opt_den.step()

                running_loss += loss.item()

                # Log statistics every `print_freq` iterations
                # if i % print_freq == print_freq - 1 or i == len(self.train_dl) - 1:
                #     with torch.no_grad():
                #         mse_y = self.criterion(self.net.decoder(code + noise), x)
                #         mse_z_star = self.criterion(self.net.decoder(code), x)
                #         mse_dn = self.criterion(self.net.decoder(code + noise - residual), x)

                #     log_message = "[{:4d}, {:5d}] loss: {:.5f}, MSE y: {:.5f}, MSE z*: {:.5f}, MSE denoised: {:.5f}".format(
                #         epoch + 1, i + 1, running_loss / (i + 1), mse_y.item(), mse_z_star.item(), mse_dn.item())
                #     print(log_message)

            avg_loss = running_loss / len(self.train_dl)
            self.writer.add_scalar("train/denoiser_loss", avg_loss, epoch)
            print(f"[Epoch {epoch:03d}] Denoiser Loss: {avg_loss:.4f}")
            self.save_model(epoch, self.model)

        self.writer.close()
        self.save_config()

    # def evaluate_epoch(self):
    #     self.net.eval()
    #     self.model.eval()
    #     total_loss = 0.0
    #     batches = 0

    #     with torch.no_grad():
    #         for batch in self.test_dl:
    #             images = batch[0] if isinstance(batch, (tuple, list)) else batch
    #             images = images.to(self.device)

    #             code = self.net.encoder(images)
    #             noise = torch.randn_like(code) * self.stddev
    #             y     = code + noise
    #             residual = self.model(y)

    #             loss = self.criterion(residual, noise)
    #             total_loss += loss.item()
    #             batches += 1

    #     return total_loss / batches
