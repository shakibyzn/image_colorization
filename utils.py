import argparse
import os
import random
import shutil

import numpy as np
import torch
from skimage import io
from skimage.color import rgb2lab
from skimage.transform import resize
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--train_size', type=int, default=30_500, help='number of training images')
    parser.add_argument('--valid_size', type=int, default=3_000, help='number of validation images')
    parser.add_argument('--test_size', type=int, default=3_000, help='number of test images')
    parser.add_argument('--patience', type=int, default=8, help='patience for early stopping')
    parser.add_argument('--early_stop', type=bool, default=True, help='early stopping')
    parser.add_argument('--scheduler', type=bool, default=True, help='scheduler')
    parser.add_argument('--wandb', type=bool, default=False, help='wandb')
    parser.add_argument('--wandb_name', type=str, default="koalarization", help='wandb name')
    parser.add_argument("--wandb_key",
                        type=str,
                        default="dfe5edf7e2dc6ae6b448271cfa1093fad2c9d433",
                        help="enter your wandb key if you didn't set on your os")  # Maryam's key
    parser.add_argument('--portion', type=float, default=0.0, help='portion of data to be used')
    args = parser.parse_args()
    return args


def load_data(args):
    if not os.path.exists('dataset'):
        for name in ['train', 'validation', 'test']:
            path = "dataset/" + name + "/"
            os.makedirs(path, exist_ok=True)

        # Setting up Breakpoints
        num_train = args.train_size
        num_val = args.valid_size
        num_test = args.test_size

        base_dir = 'val_256/'
        files = os.listdir(base_dir)
        index = 0
        for image in files:

            test = io.imread(base_dir + image)
            if test.ndim != 3:
                continue

            # Pick what folder to place image into
            if index < num_train:
                shutil.copyfile(base_dir + image, "dataset/train/" + image)
            elif index < (num_train + num_val):
                shutil.copyfile(base_dir + image, "dataset/validation/" + image)
            elif index < (num_train + num_val + num_test):
                shutil.copyfile(base_dir + image, "dataset/test/" + image)
            else:
                break
            index += 1

        # Get Dataset Objects
        train_images = ModelDataset(base_dir="./dataset/train")
        val_images = ModelDataset(base_dir="./dataset/validation")
        test_images = ModelDataset(base_dir="./dataset/test")

    else:
        # Get Dataset Objects
        train_images = ModelDataset(base_dir="./dataset/train")
        val_images = ModelDataset(base_dir="./dataset/validation")
        test_images = ModelDataset(base_dir="./dataset/test")
        if args.portion != 0:
            train_images, _ = torch.utils.data.random_split(train_images, [int(args.portion * len(train_images)),
                                                     len(train_images) - int(args.portion * len(train_images))])

            val_images, _ = torch.utils.data.random_split(val_images, [int(args.portion * len(val_images)),
                                                    len(val_images) - int(args.portion * len(val_images))])

            test_images, _ = torch.utils.data.random_split(test_images, [int(args.portion * len(test_images)),
                                                     len(test_images) - int(args.portion * len(test_images))])

    return train_images, val_images, test_images


class ModelDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.all_imgs = os.listdir(base_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):

        # Import Image and get LAB Image
        img_name = os.path.join(self.base_dir, self.all_imgs[idx])
        image = io.imread(img_name)
        image  = image / 255.0
        lab = rgb2lab(image)

        # L Channel
        L = np.expand_dims(lab[:,:,0], axis=2)
        L /= 50.07
        L -= 1.0

        # AB Channels
        AB = lab[:,:,1:].transpose(2,0,1).astype(np.float32)
        AB /= 128.0

        # Scale For Inception
        L_inc = resize(np.repeat(L,3,axis=2), (299, 299)).transpose(2,0,1).astype(np.float32)

        # Scale for Encoder
        L_enc = resize(np.repeat(L,3,axis=2), (256, 256)).transpose(2,0,1).astype(np.float32)

        # Build Sample Dict.
        sample = {"L":L.transpose(2,0,1).astype(np.float32),
                 "L_inc":L_inc, "L_enc":L_enc, "AB":AB,
                 "RGB": image}

        return sample


def train(trainloader, model, inception_model, optimizer, criterion, scheduler, device, args):
    model.train()
    total_loss = 0
    for data in trainloader:
        enc_in = data["L_enc"].to(device)
        inc_in = data["L_inc"].to(device)
        AB = data["AB"].to(device)
        L = data["L"].to(device)

        # Get Inception Output
        out_incept, _ = inception_model(inc_in)
        # Get Network AB
        net_AB = model(enc_in, out_incept)
        loss = criterion(net_AB, AB)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.scheduler:
            scheduler.step()

    if args.wandb:
        wandb.log({"Training Loss": total_loss / len(trainloader)})
    else:
        print("Training Loss:", total_loss / len(trainloader))


def validate(validloader, model, inception_model, criterion, device, args, is_test=False):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        total_ssim = 0
        loss_type = "Test" if is_test else "Validation"
        i = 0
        for data in validloader:
            enc_in = data["L_enc"].to(device)
            inc_in = data["L_inc"].to(device)
            AB = data["AB"].to(device)
            L = data["L"].to(device)

            # Get Inception Output
            out_incept, _ = inception_model(inc_in)
            # Get Network AB
            net_AB = model(enc_in, out_incept)
            loss = criterion(net_AB, AB)
            total_loss += loss.item()

            if is_test:
                img = np.concatenate([L.cpu(), net_AB.cpu()], axis=1)
                # print(img.shape)
                img[:, 0, :, :] += 1.0
                img[:, 0, :, :] *= 50.07
                img[:, 1:, :, :] *= 128.0
                result_img = lab2rgb(img.transpose(0, 2, 3, 1))  # lab2rgb(img.transpose(1, 2, 0))
                real_img = data["RGB"].detach().numpy()
                # save the first image of the batch
                plt.imshow(result_img[0])
                plt.savefig(f'results/seed_{args.seed}/portion_{args.portion}/gen_{i}.png')
                plt.imshow(real_img[0])
                plt.savefig(f'results/seed_{args.seed}/portion_{args.portion}/real_{i}.png')
                i += 1
                # compute ssim for the whole batch
                total_ssim += ssim(result_img, real_img, multichannel=True)

        if args.wandb:
            wandb.log({f"{loss_type} Loss": total_loss / len(validloader)})
            if is_test:
                wandb.log({f"{loss_type} ssim": total_ssim / len(validloader)})
        else:
            print(f"{loss_type} Loss:", total_loss / len(validloader))
            if is_test:
                print({f"{loss_type} ssim": total_ssim / len(validloader)})
