import torch
import torch.nn as nn
from torchvision.models import inception_v3

import utils
import wandb
from models import deep_colorization
import os
os.environ['TORCH_HOME'] = 'models_cpt'


def main():
    args = utils.load_config()
    print(args)
    # set seed
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb
    if args.wandb:
        run = wandb.init(project=args.wandb_name, entity='clean_label_poisoning_attack')
        wandb.config.update(args)

    if not os.path.exists('dataset'):
        utils.load_data(args)

    # Get Dataset Objects
    train_images = utils.ModelDataset(base_dir="./dataset/train")
    val_images = utils.ModelDataset(base_dir="./dataset/validation")
    test_images = utils.ModelDataset(base_dir="./dataset/test")

    # Pass Into Loaders
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=args.batch_size, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_images, batch_size=args.batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=args.batch_size, num_workers=2)

    # load inception_v3 and primary models
    inception_model = inception_v3(pretrained=True).to(device)
    model = deep_colorization.ColorNet().to(device)

    # optimizer, criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # scheduler
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # training
    for epoch in range(args.epochs):
        print("Epoch", epoch + 1)
        utils.train(train_loader, model, inception_model, optimizer, criterion, scheduler, device, args)
        utils.validate(val_loader, model, inception_model, criterion, device, args, is_test=False)

    # evaluation
    utils.validate(test_loader, model, inception_model, criterion, device, args, is_test=True)


if __name__ == '__main__':
    main()
