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

class SumTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model = Generator(args).to(self.device)
        self.denoiser = BF_CNN(1,64,3,20,args.num_channels).to(self.device)
        self.opt_jscc = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.opt_den  = torch.optim.Adam(self.denoiser.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss(reduction="sum")
        self.stddev = 10**(-0.05*args.base_snr)

    def train(self):
        for epoch in range(self.args.out_e):
            # ---- 1) Train JSCC ----
            self.model.train();   self.denoiser.eval()
            jscc_loss = 0
            for x, _ in self.train_dl:
                x = x.to(self.device)
                rec = self.model(x, self.stddev)
                loss = self.criterion(rec, x)
                self.opt_jscc.zero_grad(); loss.backward(); self.opt_jscc.step()
                jscc_loss += loss.item()
            avg_jscc = jscc_loss / len(self.train_dl)

            # ---- 2) Train BF-CNN ----
            self.model.eval();   self.denoiser.train()
            den_loss = 0
            for x, _ in self.train_dl:
                x = x.to(self.device)
                with torch.no_grad():
                    z = self.model.encoder(x)
                noise = torch.randn_like(z)*self.stddev
                res = self.denoiser(z+noise)
                loss = self.criterion(res, noise)
                self.opt_den.zero_grad(); loss.backward(); self.opt_den.step()
                den_loss += loss.item()
            avg_den = den_loss / len(self.train_dl)

            # ---- Log & Save ----
            self.writer.add_scalar("loss/jscc", avg_jscc, epoch)
            self.writer.add_scalar("loss/denoiser", avg_den, epoch)
            print(f"Epoch {epoch}: JSCC {avg_jscc:.4f}, Den {avg_den:.4f}")
            self.save_model(epoch, self.model)
            self.save_model(epoch, self.denoiser)

        self.writer.close()
        self.save_config()


    # def evaluate_epoch(self):
    #     stddev = 10 ** (-0.05 * self.base_snr)  # Tính stddev từ base_snr
    #     self.model.eval()
    #     epoch_loss = 0

    #     with torch.no_grad():
    #         for iter, batch in enumerate(self.test_dl):
    #             if isinstance(batch, (tuple, list)) and len(batch) == 2:
    #                 images, _ = batch
    #             else:
    #                 images = batch
    #             images = images.to(self.device)
    #             model_out = self.model(images, stddev)  # Truyền stddev vào mô hình
    #             loss = self.criterion(images, model_out)
    #             epoch_loss += loss.item()

    #     return epoch_loss