# from tqdm import tqdm
# import numpy as np
# import os
# import glob
# import time

# import torch
# from torch import nn
# from torch.optim import Adam

# from train.train_base import BaseTrainer
# from models.swinjscc import *
# #from modules.distortion import Distortion

# import torch
# from torch.optim import Adam
# from tqdm import tqdm

# #from models.swinjscc import SWINJSCC
# from models.eDjscc import *
# #from losses import Distortion  # hoặc nơi bạn định nghĩa loss của SwinJSCC

# class eDJSCCTrainer(BaseTrainer):
#     def __init__(self, args):
#         super().__init__(args)
#         # 1) Deep JSCC generator
#         self.model = Generator(args).to(self.device)
#         # 2) Bias‐free denoiser
#         self.denoiser = BF_CNN(1, 64, 3, 20, 6).to(self.device)
#         # 3) 2 optimizer
#         self.opt_jscc = Adam(self.model.parameters(), lr=args.lr)
#         self.opt_den  = Adam(self.denoiser.parameters(), lr=args.lr)
#         self.criterion = nn.MSELoss(reduction="sum")
#         self.stddev = 10 ** (-0.05 * args.base_snr)
#         #self.writer = SummaryWriter(self.root_log_dir)

#     def train(self):
#         for epoch in range(self.args.out_e):
#             epoch_loss_jscc = 0.0
#             epoch_loss_den  = 0.0
#             # —— 1) Train denoiser only —— #
            

#             # —— 2) Train JSCC generator only —— #
#             self.model.train()
#             #self.denoiser.eval()
#             for x, _ in tqdm(self.train_dl, desc=f"JSCC Ep{epoch}"):
#                 x = x.to(self.device)
                
#                     # z = self.model.encoder(x)
#                     # noise = torch.randn_like(z) * self.stddev
#                     # y = z + noise
#                     # res = self.denoiser(y)
#                     # y_dn = y - res
#                 rec = self.model(x, self.stddev)  # Truyền stddev vào mô hình
#                 loss_jscc = self.criterion(x, rec)
#                 # Backward
#                 self.opt_jscc.zero_grad()
#                 loss_jscc.backward()   
#                 self.opt_jscc.step()
#                 epoch_loss_jscc += loss_jscc.detach().item()
            
#             self.model.eval()
#             self.denoiser.train()
#             for x, _ in tqdm(self.train_dl, desc=f"Denoiser Ep{epoch}"):
#                 x = x.to(self.device)
               
#                 z = self.model.encoder(x)
#                 noise = torch.randn_like(z) * self.stddev
#                 y = z + noise 
#                 residual = self.denoiser(y)
#                 loss_codeword = self.criterion(residual, noise)
#                 self.opt_den.zero_grad()
#                 loss_codeword.backward()
#                 self.opt_den.step()
#                 epoch_loss_den += loss_codeword.detach().item()

#             # log & save
#             avg_den = epoch_loss_den  / len(self.train_dl)
#             avg_jsc = epoch_loss_jscc / len(self.train_dl)
#             self.writer.add_scalar("train/denoiser_loss", avg_den, epoch)
#             self.writer.add_scalar("train/jscc_loss",     avg_jsc, epoch)
#             print(f"[Epoch {epoch}] Denoiser: {avg_den:.4f}, JSCC: {avg_jsc:.4f}")
#             self.save_model(epoch, self.model)
#             self.save_model(epoch, self.denoiser)

#         #self.writer.close()
#         self.writer.close()
#         self.save_config()


#     def evaluate_epoch(self):
#         stddev = 10 ** (-0.05 * self.base_snr)  # Tính stddev từ base_snr
#         self.model.eval()
#         epoch_loss = 0

#         with torch.no_grad():
#             for iter, batch in enumerate(self.test_dl):
#                 if isinstance(batch, (tuple, list)) and len(batch) == 2:
#                     images, _ = batch
#                 else:
#                     images = batch
#                 images = images.to(self.device)
#                 model_out = self.model(images, stddev)  # Truyền stddev vào mô hình
#                 loss = self.criterion(images, model_out)
#                 epoch_loss += loss.item()

#         return epoch_loss

    # def test(self):
    #     # Gọi evaluate từ BaseTrainer
    #     self.evaluate(self.args.config_path, self.args)
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.optim import Adam

from train.train_base import BaseTrainer
import torch
from torch.optim import Adam
import torch.optim as optim
from tqdm import tqdm
from utils.data_utils import image_normalization
#from models.swinjscc import SWINJSCC
from models.eDjscc import *
#from losses import Distortion  # hoặc nơi bạn định nghĩa loss của SwinJSCC

class eDJSCCTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        # Khởi tạo model SwinJSCC với args
        self.model = Generator(args).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr,betas=(0.0, 0.9))
        if 'cifar10' in args.ds:
            self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.out_e // 2, gamma=0.1)
        self.criterion = nn.MSELoss(reduction="mean") 
        #self.bfcnn = BF_CNN(1, 64, 3, 20, 6) # num_channels = 6
        self.base_snr = args.base_snr

    def train(self):
        stddev = 10 ** (-0.05 * self.base_snr)  # Tính stddev từ base_snr

        for epoch in range(self.args.out_e):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(self.train_dl, desc=f"Epoch {epoch}"):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch
                images = images.to(self.device)
                # Forward
                rec = self.model(images, stddev)  # Truyền stddev vào mô hình
                mse_loss = self.criterion(images, rec)
                loss = mse_loss 
                # Backward
                self.optimizer.zero_grad()
                loss.backward()   
                self.optimizer.step()
                total_loss += loss.item()
                # if i % print_freq == print_freq - 1 or i == len(self.train_dl) - 1

                #     print(f"[Train] Epoch {epoch}, Iter {i}: loss = {loss.item():.4f}")
            avg_loss = total_loss / len(self.train_dl) 
            self.writer.add_scalar('train/loss', avg_loss, epoch)
            print(f"[Train] Epoch {epoch}: loss = {avg_loss:.4f}")

            # Validation
            val_loss = self.evaluate_epoch()/len(self.test_dl)
            self.writer.add_scalar('val/loss', val_loss, epoch)
            print(f"[Val]   Epoch {epoch}: loss = {val_loss:.4f}")

            self.save_model(epoch=epoch, model=self.model)
            #self.scheduler_G.step()



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
    
    def evaluate_epoch(self):
        stddev = 10 ** (-0.05 * self.base_snr)  # Tính stddev từ base_snr
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for iter, batch in enumerate(self.test_dl):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch
                images = images.to(self.device)
                model_out = self.model(images, stddev)  # Truyền stddev vào mô hình
                loss = self.criterion(images, model_out)
                epoch_loss += loss.item()

        return epoch_loss



        #eturn epoch_loss
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
    #             model_out = self.model(images)  # Truyền stddev vào mô hình
    #             loss = self.criterion(images, model_out)
    #             epoch_loss += loss.item() 
    
    
    
    # def test(self):
    #     # Gọi evaluate từ BaseTrainer
    #     self.evaluate(self.args.config_path, self.args)





