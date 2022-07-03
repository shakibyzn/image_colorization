from skimage import io
import os


def set_seed():
    pass


def load_config():
    # argparse
    # epoch, lr, wandb, scheduler, seed, early_stop, patience, train_samples
    pass


def load_data():
    for name in ['train', 'validation','test']:
        path = "dataset/" + name + "/"
    os.makedirs(path, exist_ok=True)

    # Setting up Breakpoints
    num_train = 2_000 # using argparse
    num_val = 500
    num_test = 1_000

    base_dir = './val_256/'
    files = os.listdir(base_dir)
    index = 0
    for i,image in enumerate(files):

        test = io.imread(base_dir + image)
        if test.ndim != 3:
            continue

        # Pick what folder to place image into
        if i < num_train:
            os.rename(base_dir + image, "./dataset/train/" + image)
        elif i < (num_train + num_val):
            os.rename(base_dir + image, "./dataset/validation/" + image)
        elif i < (num_train + num_val + num_test):
            os.rename(base_dir + image, "./dataset/test/" + image)
        else:
            break
        index += 1