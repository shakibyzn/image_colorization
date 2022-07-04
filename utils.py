import argparse
import os
import random

import numpy as np
import torch
import wandb
from skimage import io
from skimage.color import rgb2lab
from skimage.transform import resize


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
    parser.add_argument('--train_size', type=int, default=5_000, help='number of training images')
    parser.add_argument('--patience', type=int, default=8, help='patience for early stopping')
    parser.add_argument('--early_stop', type=bool, default=True, help='early stopping')
    parser.add_argument('--scheduler', type=bool, default=True, help='scheduler')
    parser.add_argument('--wandb', type=bool, default=False, help='wandb')
    args = parser.parse_args()
    return args


def load_data(args):
    for name in ['train', 'validation', 'test']:
        path = "dataset/" + name + "/"
        os.makedirs(path, exist_ok=True)

    # Setting up Breakpoints
    num_train = args.train_size
    num_val = 200
    num_test = 200

    base_dir = 'val_256/'
    files = os.listdir(base_dir)
    index = 0
    for image in files:

        test = io.imread(base_dir + image)
        if test.ndim != 3:
            continue

        # Pick what folder to place image into
        if index < num_train:
            os.rename(base_dir + image, "dataset/train/" + image)
        elif index < (num_train + num_val):
            os.rename(base_dir + image, "dataset/validation/" + image)
        elif index < (num_train + num_val + num_test):
            os.rename(base_dir + image, "dataset/test/" + image)
        else:
            break
        index += 1


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


def validate(validloader, model, inception_model, criterion, device, args):
    with torch.no_grad():
        model.eval()
        total_loss = 0
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

        if args.wandb:
            wandb.log({"Validation Loss": total_loss / len(validloader)})
        else:
            print("Validation Loss:", total_loss / len(validloader))
