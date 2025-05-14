import numpy as np
import torch
import torch.nn as nn
from models.model_base import BaseModel
class UnitNorm(nn.Module):
    def __init__(self, max_power=1, eps=1e-5):
        super(UnitNorm, self).__init__()
        self.max_power = max_power

    def forward(self, x):
        x_shape = x.size()
        x = x.reshape(x_shape[0], -1)
        multiplier = self.max_power * np.sqrt(x.size(1))
        proj_idx = torch.norm(x, p=2, dim=1) > multiplier
        x[proj_idx] = multiplier * x[proj_idx] / torch.norm(x[proj_idx], p=2, dim=1, keepdim=True)
        return x.reshape(*x_shape)


def unitnorm(x, max_power=1.0):
	x_shape = x.size()
	x = x.reshape(x_shape[0], -1)
	multiplier = max_power * np.sqrt(x.size(1))
	proj_idx = torch.norm(x, p=2, dim=1) > multiplier
	x[proj_idx] = multiplier * x[proj_idx] / torch.norm(x[proj_idx], p=2, dim=1, keepdim=True)
	return x.reshape(*x_shape)


class ResBlock(nn.Module): # dùng residual block, tránh hiện tượng vanishing grad 
    def __init__(self, dim1, dim2, normalization=nn.BatchNorm2d, activation=nn.ReLU, r=2, conv=None, kernel_size=5):
        super(ResBlock, self).__init__()
        # Gồm 2 nhánh nhánh một qua 2 block 1, 2 nhánh 2 qua shortcut (giữ nguyên) rồi cộng lại 
        if conv is None: 
            conv = nn.Conv2d # dùng conv2d 
        """Block 1 gômf conv -> norm -> act(giảm số chiều biểu diễn ẩn đi r lần) 
        và block 2 gồm conv -> norm(số chiều từ về dim2(output channel)) """ 
        self.block1 = nn.Sequential(conv(dim1, dim1//r, kernel_size, stride=1, padding=(kernel_size-1)//2), normalization(dim1//r), activation())
        self.block2 = nn.Sequential(conv(dim1//r, dim2, kernel_size, stride=1, padding=(kernel_size-1)//2), normalization(dim2))
        if dim1 != dim2: # nếu dim1 khác dim2 thì shortcut sẽ tích chập chuyển từ dim1 -> dim2 để cùng đầu ra với 2 block còn cộng
            self.shortcut = conv(dim1, dim2, 1, bias=False)
        else:
            self.shortcut = nn.Identity() # giữ nguyên
    def forward(self, x):
        x_short = self.shortcut(x) # qua một lớp tích chập or giữ nguyên(indentity)
        x = self.block2(self.block1(x)) # một nhánh qua 2 block
        return x+x_short # B, dim2, H, W


class Encoder_CIFAR(nn.Sequential):
    """Encoder Module x -> conv(stride 1, kernel = 7) -> 1 số Conv_block(mỗi bước lại nhân đôi số kênh (<max), kernel =3, stride = 2) -> norm +act
    -> 1 số Res_block(có thể có conv nhưng giữ kích thước) -> Res_block với số kênh đầu ra num_out còn encoder bên trên thì lớp cuối là conv"""
    def __init__(self, 
                 num_out,
                 num_hidden, 
                 num_conv_blocks=2,
                 num_residual_blocks=2, # ít hơn bên trên 1 khối
                 conv = nn.Conv2d,
                 normalization=nn.BatchNorm2d, 
                 activation=nn.PReLU, 
                 power_norm="hard", 
                 primary_latent_power=1.0,
                 r=2,
                 max_channels=512,
                 first_conv_size=7,
                 conv_kernel_size=3, 
                 residual=True,
                 **kwargs):

        bias = normalization==nn.Identity # kiểm tra xem norm có phải là identity không, nếu không có chuẩn hóa thì bật bias = true 
        layers = [conv(3, num_hidden, first_conv_size, stride=1, padding=(first_conv_size-1)//2, bias=bias),
                    normalization(num_hidden),
                    activation()]

        channels = num_hidden
        for _ in range(num_conv_blocks): # số làn chia 2 giảm kích thước vẫn thế  = num_conv (bên trên có 1 lần giảm bên ngoài rồi nên chỉ)
            channels *= 2   # mỗi bước đều gấp đôi só kênh, dim1(in) = channel //2 va dim2(out) =channel(nhân đôi qua từng bước)
            layers += [conv(np.minimum(channels//2, max_channels), np.minimum(channels, max_channels), 3, stride=2, padding=1, bias=bias)]
            layers += [normalization(np.minimum(channels, max_channels)),
                       activation()]
            if residual:
                layers += [ResBlock(np.minimum(channels, max_channels), np.minimum(channels, max_channels), normalization, activation, conv=conv, r=r, kernel_size=3), activation()]
            else:
                layers += [conv(np.minimum(channels, max_channels), np.minimum(channels, max_channels), 3, stride=1, padding=1), normalization(np.minimum(channels, max_channels)), activation()]

        for _ in range(num_residual_blocks): 
            if residual:
                layers += [ResBlock(np.minimum(channels, max_channels), np.minimum(channels, max_channels), normalization, activation, conv=conv, r=r, kernel_size=3), activation()]
            else:
                layers += [conv(np.minimum(channels, max_channels),np.minimum(channels, max_channels), conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2), 
                        normalization(np.minimum(channels, max_channels)), 
                        activation()]

        if residual:
            layers += [ResBlock(np.minimum(channels, max_channels), num_out, normalization, activation, conv=conv, r=r, kernel_size=3)]
        else:
            layers += [conv(np.minimum(channels, max_channels), num_out, 3, stride=1, padding=1)]
        if power_norm == "hard":
            layers += [UnitNorm()]
        elif power_norm == "soft":
            layers += [nn.BatchNorm2d(num_out, affine=False)]
        elif power_norm == "none":
            pass
        else:
            raise NotImplementedError()

        super(Encoder_CIFAR, self).__init__(*layers)


class Decoder_CIFAR(nn.Sequential):
    def __init__(self, 
                num_in, 
                num_hidden, # vẫn giống bên encode
                num_conv_blocks=2,
                num_residual_blocks=2,
                normalization=nn.BatchNorm2d, 
                activation=nn.PReLU, 
                no_tanh=False,
                bias_free=False,
                r=2,
                residual=True,
                max_channels=512,
                last_conv_size=5,
                normalize_first=False,
                conv_kernel_size=3,
                **kwargs):

        channels = num_hidden * (2**num_conv_blocks) # là channel đầu vào của decoder Cifar do bên encoder num_hidden được nhân đôi num_conv lần 

        layers = [nn.Conv2d(num_in, min(max_channels, channels), 3, stride=1, padding=1, bias=False),
                  normalization(channels),
                  activation()] 

        for _ in range(num_residual_blocks):
            if residual:
                layers += [ResBlock(min(max_channels, channels), min(max_channels, channels), normalization, activation, r=r, kernel_size=conv_kernel_size), activation()]
            else:
                layers += [nn.Conv2d(min(max_channels, channels), min(max_channels, channels), conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2), normalization(channels), activation()]

        for _ in range(num_conv_blocks):
            channels = channels // 2
            layers += [nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                    nn.Conv2d(min(max_channels, channels*2), min(max_channels, channels), 3, 1, 1, bias=False),
                    normalization(min(max_channels, channels)),
                    activation()]
            if residual:
                layers += [ResBlock(min(max_channels, channels), min(max_channels, channels), normalization, activation, r=r, kernel_size=3), activation()]
            else:
                layers += [nn.Conv2d(min(max_channels, channels), min(max_channels, channels), conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2), normalization(channels), activation()]

        layers += [nn.Conv2d(num_hidden, 3, last_conv_size, stride=1, padding=(last_conv_size-1)//2, bias=False)]

        if not normalize_first:
            layers += [normalization(3)]
        if not no_tanh:
            layers += [nn.Tanh()]

        super(Decoder_CIFAR, self).__init__(*layers)


class Generator(nn.Module):
    def __init__(self,args):
        super(Generator, self).__init__()
        num_out = args.num_channels # số kênh đầu vào của decoder
        num_hidden = 32
        num_conv_blocks = 2
        num_residual_blocks = 2 
        power_norm = "hard"
        num_in = num_out
        encoder = Encoder_CIFAR(num_out,
                      num_hidden,
                      num_conv_blocks,
                      num_residual_blocks,
                      normalization=nn.BatchNorm2d,
                      activation=nn.PReLU,
                      power_norm="hard")
       
        decoder = Decoder_CIFAR(num_in,
                      num_hidden,
                      num_conv_blocks,
                      num_residual_blocks,
                      normalization=nn.BatchNorm2d,
                      activation=nn.PReLU,
                      no_tanh=False)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inp, stddev, return_latent=False):
        code = self.encoder(inp) # code <-> latent z, 
        #print("code size:", code.size() if code is not None else "None")
        chan_noise = torch.randn_like(code) * stddev # sttdev = 10^(-0.05 * SNR)
        y = code + chan_noise # z + n
        reconst = self.decoder(y) 
        #print("Helloooooo")
        if return_latent:
            return reconst, (y, code)
        else:
            return reconst





"""BF_CNN"""

class BF_CNN(nn.Module):

    def __init__(self, padding, num_kernels, kernel_size, num_layers, num_channels):
        super(BF_CNN, self).__init__()

        self.padding = padding
        self.num_kernels = num_kernels # trong báo là 64 
        self.kernel_size = kernel_size # trong báo là 3x3
        self.num_layers = num_layers # số conv = 20 
        self.num_channels = num_channels

        self.conv_layers = nn.ModuleList([])
        self.running_sd = nn.ParameterList([]) # parameterlist lưu lại giá trị std và gamma 
        self.gammas = nn.ParameterList([])


        self.conv_layers.append(nn.Conv2d(self.num_channels,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))# num_channel -> num_kernel kênh ẩn trung gian

        for l in range(1,self.num_layers-1):
            self.conv_layers.append(nn.Conv2d(self.num_kernels ,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))
            self.running_sd.append( nn.Parameter(torch.ones(1,self.num_kernels,1,1), requires_grad=False) )
            g = (torch.randn( (1,self.num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
            self.gammas.append(nn.Parameter(g, requires_grad=True) )

        self.conv_layers.append(nn.Conv2d(self.num_kernels,self.num_channels, self.kernel_size, padding=self.padding , bias=False))


    def forward(self, x):
        relu = nn.ReLU(inplace=True)
        x = relu(self.conv_layers[0](x)) # Conv(x) đầu tiền -> ReLU 
        """Câc lớp tích chập giữa num_kernel -> num_kernel và tính thêm var -> sqrt(var) = std và cập nhật """
        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x) 
            # BF_BatchNorm: std theo batch(0) và H(2), W(3) còn giữ chiều C -> (1,C,1,1), ứng với σ 
            sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ 1e-05) 

            if self.conv_layers[l].training:
                x = x / sd_x.expand_as(x) # nếu đang train thì chia cho sd_x đã expand -> B, C, H, W để  tự chuẩn hóa loaị bỏ bías 
                self.running_sd[l-1].data = (1-.1) * self.running_sd[l-1].data + .1 * sd_x # cập nhật giá trị sd_x vào parameterlist theo cthuc
                # 0.9 cái cũ + 0.1 cái mới
                x = x * self.gammas[l-1].expand_as(x) # scale bằng gammas

            else:
                x = x / self.running_sd[l-1].expand_as(x) # nếu không phải train thì chia luôn cho sd đã luuw chứ không cần tính mới nữa 
                x = x * self.gammas[l-1].expand_as(x)

            x = relu(x)

        x = self.conv_layers[-1](x)

        return x
#TODO: Kết hợp Generator và BF_CNN thành một model luôn
class eDJSCC(nn.Module):
    def __init__(self, args):
        super(eDJSCC, self).__init__()
        num_channels = getattr(args, 'num_channels', 16)  # Thêm giá trị mặc định
        self.net = Generator(args)
        self.encoder = Generator(args).encoder
        self.decoder = Generator(args).decoder
        self.bfcnn = BF_CNN(1, 64, 3, 20, num_channels)  # Sử dụng num_channels
        self.net.denoiser =lambda z_: -self.bfcnn(z_)
        self.iter = args.num_iter
        self.base_snr = args.base_snr
        self.stddev = 10 ** (-0.05 * args.base_snr)
        self.lr = args.lr
        self.criterion = nn.MSELoss(reduction="sum")
        """delta và alpha trong eq 8 và 9 được dùng để điều chỉnh độ lớn """
        """bước nhảy khi SNR train và test khác nhau, alpha quyết định trọng só prior, delta các điều chỉnh bước nhảy n'"""
        self.delta = 1.0 # 1 với cifar; với openimage thì delta = 0.5 1, 2 lần lượt
        self.alpha = 1.0 # nếu dùng cifar10; nếu OpenImage(CPP1/16)thì alpha=2 nếu train 1dB, 4 nếu 7, 13
    def forward(self, x):
        codeword = self.net.encoder(x)
        #print("codeword size:", codeword)
        if torch.isnan(codeword).any():
            print("[ERROR] codeword contains NaN values")
            return None
        noise = torch.randn_like(codeword) * self.stddev
        print("snr:", self.base_snr)
        print("stddev:", self.stddev)
        #print("noise size:", noise)
        y = codeword + noise
        #print("y size:", y)
        if torch.isnan(y).any():
            print("[ERROR] y contains NaN values")
            return None
        zt = y.clone().detach().requires_grad_(True)
        for i in range(self.iter):
            recon = self.net.decoder(zt)
            #print("num iter:", i)
            #print(f"Iteration {i}: rdfadsfs size:", zt if zt is not None else "dsfsadfa")
            #print(f"Iteration {i}: recon size:", recon.size() if recon is not None else "None")
            y_hat = self.net.encoder(recon)
            nll = ((y - y_hat)**2).sum() / (2 * self.stddev**2)
            zt_grad = torch.autograd.grad(nll, zt)[0]
            if torch.isnan(zt_grad).any():
                print(f"[ERROR] zt_grad contains NaN values at iteration {i}")
                return None
            dt = self.net.denoiser(zt)
            lr = self.lr / max(0.1, (self.stddev**2 / self.base_snr)**self.delta)
            zt = zt - lr * (zt_grad - self.alpha * max(0.1, (self.stddev**2 / self.base_snr)**2) * dt)
            # if torch.isnan(zt).any():
            #     print(f"[ERROR] zt contains NaN values at iteration {i}")
            #     return None
            zt = zt.detach().requires_grad_(True)
        x_hat = self.net.decoder(zt)
        #print("x_hat size:", x_hat.size() if x_hat is not None else "None")
        if torch.isnan(x_hat).any():
            print("[ERROR] x_hat contains NaN values")
            return None
        return x_hat

