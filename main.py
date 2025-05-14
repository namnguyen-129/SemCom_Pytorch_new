import os
import argparse
from train.train_djsccn import DJSCCNTrainer
from train.train_swindjscc import SWINJSCCTrainer
from train.train_eDjscc import eDJSCCTrainer
from train.train_bfcnn import BF_CNNTrainer
from train.train_kethop import SumTrainer
from torch import nn
trainer_map = {
    "djsccn": DJSCCNTrainer,
    "swinjscc": SWINJSCCTrainer,
    "eDJSCC" : eDJSCCTrainer,
    "bfcnn": BF_CNNTrainer}

# ratio_list = [1/6, 1/12]
# snr_list = [19, 13, 7, 4, 1]
ratio_list = [1/6]
snr_list = [20]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./out',
                         help="Path to save outputs")
    parser.add_argument("--ds", type=str, default='cifar10',
                        choices=['cifar10', 'DIV2K', 'mnist'],
                        help="Dataset")
    parser.add_argument("--base_snr", type=float, default=10,
                        help="SNR during train")
    parser.add_argument('--channel_type', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel')
    parser.add_argument("--recl", type=str, default='mse',
                        help="Reconstruction Loss")
    parser.add_argument("--clsl", type=str, default='ce',
                        help="Classification Loss")
    parser.add_argument("--disl", type=str, default='kl',
                        help="Invariance and Variance Loss")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Inner learning Rate")

    # Thêm tham số distortion_metric
    parser.add_argument("--distortion_metric", type=str, default="MSE",
                        choices=["MSE", "MS-SSIM"],
                        help="Distortion metric to use (MSE or MS-SSIM)")

    # Loss Setting
    parser.add_argument("--cls-coeff", type=float, default=0.5,
                        help="Coefficient for Classification Loss")
    parser.add_argument("--rec-coeff", type=float, default=1,
                        help="Coefficient for Reconstruction Loss")
    parser.add_argument("--inv-coeff", type=float, default=0.2,
                        help="Coefficient for Invariant Loss")
    parser.add_argument("--var-coeff", type=float, default=0.2,
                        help="Coefficient for Variant Loss")

    # Model Setting
    parser.add_argument("--inv-cdim", type=int, default=32,
                        help="Channel dimension for invariant features")
    parser.add_argument("--var-cdim", type=int, default=32,
                        help="Channel dimension for variant features")

    # VAE Setting
    parser.add_argument("--vae", action="store_true",
                        help="vae switch")
    parser.add_argument("--kld-coeff", type=float, default=0.00025,
                        help="VAE Weight Coefficient")

    # Meta Setting
    parser.add_argument("--bs", type=int, default=128,
                        help="#batch size")
    parser.add_argument("--wk", type=int, default=os.cpu_count(),
                        help="#number of workers")
    parser.add_argument("--out-e", type=int, default=10,
                        help="#number of epochs")
    parser.add_argument("--dv", type=int, default=0,
                        help="Index of GPU")
    parser.add_argument("--device", type=bool, default=True,
                        help="Return device or not")
    parser.add_argument("--operator", type=str, default='window',
                        help="Operator for Pycharm")

    # LOGGING
    parser.add_argument('--wandb', action='store_true',
                        help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true',
                        help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, default="SemCom-",
                        help='toggle to use wandb for online saving')
    parser.add_argument('--wandb_entity', type=str, default="scalemind",
                        help='toggle to use wandb for online saving')
    parser.add_argument("--verbose", action="store_true",
                        help="printout mode")
    parser.add_argument("--algo", type=str, default="swinjscc",
                        help="necst/djsccf mode")
    
    # RUNNING
    parser.add_argument('--train_flag', type=str, default="True",
                        help='Training mode')
    # Model size of swinjscc
    parser.add_argument('--model_size', type=str, default="small",
                        choices=['small', 'base', 'large'],
                        help='Model size of swinjscc')
    # Norm

    parser.add_argument('--norm', type=bool, default=False,
                        help='Norm layer')
    # eDJSCC, cifar 10: num_channels = 16, bs = 64, lr = 0.0002; num_conv_blocks = num_res_blocks = 2
    parser.add_argument('--num_iter', type=int, default=10,help='Number of iterations for eDJSCC')
    parser.add_argument('--num_channels', type=int, default=16, help='Number of channels')
    parser.add_argument('--num_conv_blocks', type=int, default=2, help='Number of convolutional blocks')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')
 

    args = parser.parse_args()
    if args.algo not in trainer_map:
        raise ValueError("Invalid trainer")
    
    TrainerClass = trainer_map[args.algo]
    
    "Để tính snr và channel number"
    args.snr_list = snr_list
    args.ratio = ratio_list
    args.pass_channel = True
    if args.ds == 'cifar10':
        args.image_dims = (3, 32, 32)
        args.downsample = 2
        #args.bs = 128
  

    # Kích thước latent channels
    args.channel_number = int(args.var_cdim)

    # Unpack spatial dims
    _, H, W = args.image_dims

    # Thiết lập encoder_kwargs 
    if args.ds == 'cifar10':
        args.encoder_kwargs = dict(
        img_size=(H, W), patch_size=2, in_chans=args.image_dims[0],
        embed_dims=[64, 128], depths=[2, 4], num_heads=[4, 8],
        C=args.channel_number,
        window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm,  # Sử dụng nn.LayerNorm thay vì None
        patch_norm=True
    )

    # Thiết lập decoder_kwargs
        args.decoder_kwargs = dict(
        img_size=(H, W),
        embed_dims=[128, 64], depths=[4, 2], num_heads=[8, 4],
        C=args.channel_number,
        window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm,  # Sử dụng nn.LayerNorm thay vì None
        patch_norm=True
    )
    if args.train_flag == "True":
        print("Training mode")
        for ratio in ratio_list:
            for snr in snr_list:
                args.ratio = [ratio]  # Đảm bảo args.ratio luôn là một danh sách
                args.base_snr = snr

                trainer = TrainerClass(args=args)  # Đảm bảo args chứa num_channels
                trainer.train()
    else:
        print("Evaluation mode")
        trainer = TrainerClass(args=args)  # Đảm bảo args chứa num_channels

        config_dir = os.path.join(args.out, 'configs')
        config_files = [os.path.join(config_dir, name) for name in os.listdir(config_dir)
                        if (args.ds in name or args.ds.upper() in name) and args.channel_type in name and name.endswith('.yaml')]
        output_dir = args.out

        for config_path in config_files:
            trainer.evaluate(
                config_path=config_path,
                output_dir=output_dir
            )
