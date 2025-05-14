from torchvision.datasets import CIFAR10, EMNIST, MNIST, CelebA
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch
import numpy as np
import os
from glob import glob

def get_mnist(args):
    path = os.path.join(os.getcwd(), "dataset")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_set = MNIST(root=path, train=True, download=True, transform=transform)
    train_dl = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=2)
    test_set = MNIST(root=path, train=False, download=True, transform=transform)
    test_dl = DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=2)
    valid_set = MNIST(root=path, train=False, download=True, transform=transform)
    valid_dl = DataLoader(valid_set, batch_size=args.bs, shuffle=False, num_workers=2)

    return (train_dl, test_dl, valid_dl), args


def get_emnist(args):
    path = os.path.join(os.getcwd(), "dataset")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    train_set = EMNIST(root=path, split="digits", train=True, download=True, transform=transform)
    test_set = EMNIST(root=path, split="digits", train=False, download=True, transform=transform)
    valid_set = EMNIST(root=path, split="digits", train=False, download=True, transform=transform)
    train_dl = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=1)
    test_dl = DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=1)
    valid_dl = DataLoader(valid_set, batch_size=args.bs, shuffle=False, num_workers=1)

    return (train_dl, test_dl, valid_dl), args


def get_cifar10(args):
    path = os.path.join(os.getcwd(), "dataset")
    # transform = transforms.Compose(
    #     [
    #         # transforms.RandomHorizontalFlip(),
    #         # transforms.CenterCrop(148),
    #         # transforms.Resize(64),
    #         transforms.ToTensor()
    #     ]
    # )
    if args.norm:
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        test_transform = transforms.Compose([
                transforms.ToTensor()]) 
    train_set = CIFAR10(root=path, train=True, download=True, transform=train_transform)
    train_dl = DataLoader(train_set, batch_size=args.bs, shuffle=True)
    test_set = CIFAR10(root=path, train=False, download=True, transform=test_transform)
    test_dl = DataLoader(test_set, batch_size=args.bs, shuffle=False)
    valid_set = CIFAR10(root=path, train=False, download=True, transform=test_transform)
    valid_dl = DataLoader(valid_set, batch_size=args.bs, shuffle=False)
    return (train_dl, test_dl, valid_dl), args
# def get_div2k(args):
#     base_path = "/home/namdeptrai/djscc/data/DIV2K/HR_Image_dataset"
#     """Mặc định test set là kodak"""
#     test_data_dir = ["/home/namdeptrai/djscc/data/kodak/"]
#     train_data_dir = [base_path + '/DIV2K_train_HR']
#     valid_data_dir = [base_path + '/DIV2K_valid_HR']
#     img_train = []
#     for dir in train_data_dir:
#         img_train += glob(os.path.join(dir, "*.jng"))
#         img_train += glob(os.path.join(dir, "*.png"))
#     _, im_height, im_wight = args.image_dims
#     transform_train = [
#             # transforms.RandomCrop((self.im_height, self.im_width)),
#             transforms.RandomCrop((256, 256)),
#             transforms.ToTensor()]
#     img_train_path = img_train[idx]
#     img_train = Image.open(img_train_path).convert('RGB')
#     transform_train = 
def get_div2k(args):
    base_path = "/home/namdeptrai/djscc/data/DIV2K/HR_Image_dataset/"
    """Mặc định test set là kodak"""
    test_data_dir = ["/home/namdeptrai/djscc/data/kodak/"]
    train_data_dir = [base_path + '/DIV2K_train_HR/']
    valid_data_dir = [base_path + '/DIV2K_valid_HR/']
    class HR_Image(Dataset):
        def __init__(self, args, data_dir):
            self.imgs = []
            for dir in data_dir:
                self.imgs += glob(os.path.join(dir, '*.jpg'))
                self.imgs += glob(os.path.join(dir, '*.png'))
            self.imgs.sort()
        # _, self.im_height, self.im_width = config.image_dims
        # self.crop_size = self.im_height
            self.image_dims = args.image_dims
            self.transform = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor()
            ])
        def __getitem__(self, idx):
            img_path = self.imgs[idx]
            img = Image.open(img_path)
            img = img.convert('RGB')
            transformed = self.transform(img)
            return transformed
        def __len__(self):
            return len(self.imgs)
    class DataTest(Dataset):
        def __init__(self, data_dir):
            self.data_dir = data_dir
            self.imgs = []
            for dir in self.data_dir:
                self.imgs += glob(os.path.join(dir, '*.jpg'))
                self.imgs += glob(os.path.join(dir, '*.png'))
            self.imgs.sort()


        def __getitem__(self, item):
            image_ori = self.imgs[item]
        #name = os.path.basename(image_ori)
            image = Image.open(image_ori).convert('RGB')
            self.im_height, self.im_width = image.size
            if self.im_height % 128 != 0 or self.im_width % 128 != 0:
                self.im_height = self.im_height - self.im_height % 128
                self.im_width = self.im_width - self.im_width % 128
            self.transform = transforms.Compose([
                transforms.CenterCrop((self.im_width, self.im_height)),
                transforms.ToTensor()])
            img = self.transform(image)
            return img
        def __len__(self):
            return len(self.imgs)
    train_set = HR_Image(args, train_data_dir)
    valid_set = HR_Image(args, valid_data_dir)
    test_set = DataTest(test_data_dir)
    train_dl = DataLoader(train_set, batch_size = 1, shuffle = True,drop_last = True,)
    valid_dl = DataLoader(valid_set, batch_size = 1, shuffle = False, drop_last = True) #args.bs 
    test_dl = DataLoader(test_set,batch_size = 1,shuffle = False,)
    return (train_dl, valid_dl, test_dl),args

def get_cinic10(args):
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]

    transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=(-1, 1), translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
    root_dir = os.path.join(os.getcwd(), "dataset", "CINIC-10")

    train_set = ImageFolder(root=root_dir + "/train", transform=transform)
    train_dl = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=24)
    valid_set = ImageFolder(root=root_dir + "/valid", transform=transform)
    valid_dl = DataLoader(valid_set, batch_size=args.bs, shuffle=False, num_workers=24)
    test_set = ImageFolder(root=root_dir + "/test", transform=transform)
    test_dl = DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=24)
    return (train_dl, test_dl, valid_dl), args


def get_dsprites(args):
    class DSprDS(Dataset):
        def __init__(self, split='train', seed=42):
            super().__init__()
            np.random.seed(seed)
            self.root_path = "/".join(
                os.getcwd().split("/")[:-2]) + "/dataset/dsprites/source/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

            self.img_data = np.load(self.root_path, allow_pickle=True, encoding='bytes')['imgs']
            np.random.shuffle(self.img_data)

            self.ori_len = self.img_data.shape[0]

            self.ratio_mapping = {
                "train": (0, int(self.ori_len * 0.95)),
                "valid": (int(self.ori_len * 0.95), int(self.ori_len * 0.975)),
                "test": (int(self.ori_len * 0.975), int(self.ori_len))
            }

            self.split = split
            self.ratio = self.ratio_mapping[split]
            self.data = self.img_data[self.ratio[0]:self.ratio[1]]

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            img = self.data[idx]

            torch_img = torch.from_numpy(img).unsqueeze(0)

            return torch_img.float()

    train_set = DSprDS(split='train')
    train_dl = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=24)
    test_set = DSprDS(split='test')
    test_dl = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=24)
    valid_set = DSprDS(split='valid')
    valid_dl = DataLoader(valid_set, batch_size=256, shuffle=False, num_workers=24)
    return (train_dl, test_dl, valid_dl), args


def get_celeb(args):
    path = os.path.join(os.getcwd(), "dataset")

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.ToTensor()
        ]
    )

    train_set = CelebA(root=path, split='train', download=True, transform=transform)
    train_dl = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=8)
    valid_set = CelebA(root=path, split='valid', download=True, transform=transform)
    valid_dl = DataLoader(valid_set, batch_size=args.bs, shuffle=False, num_workers=8)
    test_set = CelebA(root=path, split='test', download=True, transform=transform)
    test_dl = DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=8)

    return (train_dl, test_dl, valid_dl), args



def get_ds(args):

    ds_mapping = {
        "mnist": get_mnist,
        "emnist": get_emnist,
        "dsprites": get_dsprites,
        "cifar10": get_cifar10,
        "cinic10": get_cinic10,
        "celeba": get_celeb,
        "DIV2K": get_div2k
    }
    # if args.ds == 'DIV2K':
    #     transform = transforms.Compose([
    #         transforms.Resize((args.image_dims[1], args.image_dims[2])),
    #         transforms.ToTensor()
    #     ])
    #     train_dataset = ImageFolder(root=args.train_data_dir, transform=transform)
    #     test_dataset = ImageFolder(root=args.test_data_dir, transform=transform)

    #     # Chia tập huấn luyện thành train và valid
    #     train_size = int(0.9 * len(train_dataset))  # 90% cho train
    #     valid_size = len(train_dataset) - train_size  # 10% cho valid
    #     train_subset, valid_subset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    #     train_dl = DataLoader(train_subset, batch_size=args.bs, shuffle=True, num_workers=args.wk)
    #     valid_dl = DataLoader(valid_subset, batch_size=args.bs, shuffle=False, num_workers=args.wk)
    #     test_dl = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=args.wk)

    #     return (train_dl, test_dl, valid_dl), args

    data, args = ds_mapping[args.ds](args)

    return data, args