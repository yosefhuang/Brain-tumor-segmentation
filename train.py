import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

#from dataset import BrainSegmentationDataset as Dataset
#from my_dataset import BrainSegmentationDataset as Dataset
#from my_dataset import BrainSegmentationvalDataset as valDataset
from data import BrainSegmentationDataset as Dataset
from data import BrainSegmentationvalDataset as valDataset
from logger import Logger
from loss import DiceLoss,Gen_dice_loss,BCEDiceLoss
from transform import transforms
from unet import UNet,NestedUNet
from utils import log_images, dsc,my_dsc


def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

#    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
#    unet = NestedUNet(in_ch=Dataset.in_channels, out_ch=Dataset.out_channels)
    unet = NestedUNet()
    unet.to(device)

#    dsc_loss = Gen_dice_loss()
    dsc_loss = BCEDiceLoss()
#    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,60],gamma = 0.3)

    logger = Logger(args.logs)
    loss_train = []
    loss_valid = []

    step = 0

    for epoch in range(args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)
                    print (  'epoch: ', epoch , ' | step: ', i,   ' | loss : ', loss.cpu().detach().numpy() )

                    if phase == "valid":
                        loss_valid.append(loss.item())

                        y_pred = torch.sigmoid(y_pred)
                        y_pred_np = y_pred.detach().cpu().numpy()
#                        y_pred_np = (y_pred_np > 0.5)
                        y_pred_np = np.resize(y_pred_np, (1, 256, 256))

                        y_true_np = y_true.detach().cpu().numpy()
                        y_true_np = np.resize(y_true_np, (1, 256, 256))


                        if (np.any(y_true_np)):
                            validation_pred.append(y_pred_np)
                            validation_true.append(y_true_np)


                        if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                            if i * args.batch_size < args.vis_images:
                                tag = "image/{}".format(i)
                                num_images = args.vis_images - i * args.batch_size
                                logger.image_list_summary(
                                    tag,
                                    log_images(x, y_true, y_pred)[:num_images],
                                    step,
                                )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                    )
                )
                logger.scalar_summary("val_dsc", mean_dsc, step)
                print ('best_score: ', best_validation_dsc, ' | mean_score: ',mean_dsc)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid = []

        scheduler.step()
    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    )
    valid = valDataset(
        images_dir=args.images,
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
    )
    return train, valid

#def _dice_coefficient(self, predicted, target):
#        """Calculates the Sørensen–Dice Coefficient for a
#        single sample.
#        Parameters:
#            predicted(torch.Tensor): Predicted single output of the network.
#                                    Shape - (Channel,Height,Width)
#            target(torch.Tensor): Actual required single output for the network
#                                    Shape - (Channel,Height,Width)
#        Returns:
#            coefficient(torch.Tensor): Dice coefficient for the input sample.
#                                        1 represents high similarity and
#                                        0 represents low similarity.
#        """
#    smooth = 1
#    product = torch.mul(predicted, target)
#    intersection = product.sum()
#    coefficient = (2*intersection + smooth) / (predicted.sum() + target.sum() + smooth)
#    return coefficient


def my_dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list



def dsc_per_volume(validation_pred, validation_true):
    dsc_list = []
    index = 0
    for p in range(len(validation_pred)):
        y_pred = np.array(validation_pred[p])
        y_true = np.array(validation_true[p])
        dsc_list.append(my_dsc(y_pred, y_true))
    return dsc_list




def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weight_2", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
#        "--images", type=str, default="./kaggle_train", help="root folder with images"
        "--images", type=str, default="./kaggle_train", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    args = parser.parse_args()
    main(args)
