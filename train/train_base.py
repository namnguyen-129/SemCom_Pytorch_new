import os
# from tqdm import tqdm
import numpy as np

from dataset import *
from glob import glob
import time
import json
import yaml
from tensorboardX import SummaryWriter

import torch
from torch import nn

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from utils.metric_utils import get_psnr, view_model_param
from utils.data_utils import image_normalization
from channels.channel_base import Channel
from dataset.getds import get_ds

class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.params = vars(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device else "cpu")
        self.parallel = False
        if args.train_flag == "True":
            self._setup_dirs()
        self._setup_model()
        self.times = 10
        self.channel_type = args.channel_type

        self.dataset_name = args.ds

        self.batch_size = args.bs
        self.num_workers = args.wk

    def _setup_dirs(self):
        out_dir = self.args.out
        
        phaser = str(self.args.ds).upper() + '_' + str(self.args.base_snr) + '_' + str(self.args.ratio) + '_' + \
                 str(self.args.channel_type) + '_' + str(self.args.algo) + '_channels_' + str(self.args.num_channels) + \
                 '_bs_' + str(self.args.bs) + '_' + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        
        self.root_log_dir = out_dir + '/' + 'logs/' + phaser
        self.root_ckpt_dir = out_dir + '/' + 'checkpoints/' + phaser
        self.root_config_dir = out_dir + '/' + 'configs/' + phaser
        self.writer = SummaryWriter(log_dir=self.root_log_dir)

        # Lọc các đối tượng không tuần tự hóa được
        def filter_non_serializable(obj):
            if isinstance(obj, (str, int, float, bool, list, dict, type(None))):
                return obj
            return str(obj)  # Chuyển các đối tượng không tuần tự hóa được thành chuỗi

        filtered_params = {k: filter_non_serializable(v) for k, v in self.params.items()}

        # In nội dung của filtered_params để kiểm tra
        print("Filtered Params:")
        for key, value in filtered_params.items():
            print(f"{key}: {value} ({type(value)})")

        try:
            self.writer.add_text('config', json.dumps(filtered_params, indent=4))
        except TypeError as e:
            print(f"Error serializing params: {e}")
            print(f"Filtered params: {filtered_params}")
    
    def _setup_model(self):
        
        (self.train_dl, self.test_dl, self.valid_dl), self.args = get_ds(self.args)

        if self.args.ds == "mnist":
            self.in_channel = 1  
        else:
            self.in_channel = 3
        self.class_num = 10

    def evaluate_epoch(self):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for iter, batch in enumerate(self.test_dl):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch
                images = images.cuda() if self.parallel and torch.cuda.device_count(
                ) > 1 else images.to(self.device)
                model_out = self.model(images)
                if isinstance(model_out, tuple):
                    outputs = model_out[0]
                else:
                    outputs = model_out
                loss = self.criterion(images, outputs) if not self.parallel else self.criterion(
                    images, outputs)
                epoch_loss += loss.detach().item()
            epoch_loss /= (iter + 1)

        return epoch_loss

    def evaluate(self, config_path, output_dir):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
            params = config['params']
        
        name = os.path.splitext(os.path.basename(config_path))[0]
        writer = SummaryWriter(os.path.join(output_dir, 'eval', name))
        pkl_list = glob(os.path.join(output_dir, 'checkpoints', name, '*.pkl'))
        self.model.load_state_dict(torch.load(pkl_list[-1]))        # Check later
        print("evaluate")
        self.eval_snr(writer)
        writer.close()

    def eval_snr(self, writer):
        snr_list = range(0, 26, 1)
        for snr in snr_list:
            print(f"channel: {self.channel_type} || snr: {snr}")
            self.model.change_channel(self.channel_type, snr)
            test_loss = 0
            for i in range(self.times):
                test_loss += self.evaluate_epoch()     # Check later

            test_loss /= self.times
            psnr = get_psnr(image=None, gt=None, mse=test_loss)
            writer.add_scalar('psnr', psnr, snr)

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        return loss
    
    def save_config(self):
        if not os.path.exists(os.path.dirname(self.root_config_dir)):
            os.makedirs(os.path.dirname(self.root_config_dir))
        with open(self.root_config_dir + '.yaml', 'w') as f:
            dict_yaml = {'dataset_name': self.args.ds, 'params': self.params,
                        'total_parameters': view_model_param(self.model)}
            yaml.dump(dict_yaml, f)

        # del self.model, self.optimizer, self.train_dl, self.test_dl
        # del self.writer

    def save_model(self, epoch, model):
        # Lấy tên dataset, thuật toán và số lượng kênh
        dataset_name = self.args.ds.upper() if hasattr(self.args, 'ds') else "DATASET"
        algo_name = self.args.algo if hasattr(self.args, 'algo') else "model"
        num_channels = self.args.num_channels if hasattr(self.args, 'num_channels') else "channels"

        # Tạo đường dẫn lưu checkpoint
        if not os.path.exists(self.root_ckpt_dir):
            os.makedirs(self.root_ckpt_dir)
        checkpoint_path = os.path.join(self.root_ckpt_dir, f"{dataset_name}_{algo_name}_channels_{num_channels}_snr_{self.args.base_snr}_epoch_{epoch}.pkl")
        
        # Lưu checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")