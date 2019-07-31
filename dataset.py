import os
import random

import numpy as np
import cv2
import torch
import PIL.Image as Image
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms as T
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rescale, rotate,resize
from torchvision.transforms import Compose

from utils import crop_sample, pad_sample, resize_sample, normalize_volume


def random_mask(image,mask):
    r = image[...,0]
    g = image[...,1]
    b = image[...,2]

    roi = g[mask == 255]
    unique = np.unique(roi)
    max_size = np.max(unique)
    if max_size > 90:
        g[mask == 255] -= np.random.random_integers(0,25)

    img = cv2.merge([r,g,b])
    return img, mask



def elastic_transform(image, image_mask, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
#    image = np.squeeze(image, axis=(3,))
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    image_mask = cv2.warpAffine(image_mask, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
#    image = image[..., np.newaxis]
#    print (random_state.rand(*shape))

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    img = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    mask = map_coordinates(image_mask, indices, order=1, mode='reflect').reshape(shape)
    return img,mask[..., 1]




class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=10,
        seed=42,
    ):
        assert subset in ["all", "train", "validation"]

        # read images
        self.images = []
        self.masks = []
        self.image_size = 256
        print("reading {} images...".format(subset))
        images_dir = images_dir+'/train'
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                filter(lambda f: ".tif" in f, filenames),
                key=lambda x: int(x.split(".")[-2].split("_")[4]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                     self.masks.append(filepath)
                else:
                     self.images.append(filepath)

        self.random_sampling = random_sampling
        self.transform = transform
        self.image_true = None
        self.mask_true = None
        self.count = 1


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(imread(self.images[idx]))
        mask = np.array(imread(self.masks[idx],as_gray=True))

        unique = np.unique(mask)


        if unique.shape[0] == 2:
            self.image_true = image
            self.mask_true = mask
            mask = cv2.merge([mask,mask,mask])
            image, mask = elastic_transform(image, mask, image.shape[1] * 3, image.shape[1] * 0.08, image.shape[1] * 0.08)
#            image, mask = random_mask(image, mask)
        elif unique.shape[0] == 1 and self.image_true is not None and (self.count % 3) != 0:
            image = self.image_true
            mask = self.mask_true
            mask = cv2.merge([mask,mask,mask])
            image, mask = elastic_transform(image, mask, image.shape[1] * 3, image.shape[1] * 0.08, image.shape[1] * 0.08)
#            image, mask = random_mask(image, mask)
            self.count += 1 
            if self.count >= 30001 :
                self.count = 1
        else:
            self.count += 1
            

        image = image[np.newaxis,...]
        mask = mask[np.newaxis,...]

        volumes = [image, mask]
        volumes = resize_sample(volumes, size = 256)
        image, mask = volumes

        image = normalize_volume(image)
        mask = mask[..., np.newaxis]

        image = np.squeeze(image, axis= 0)
        mask = np.squeeze(mask, axis= 0)
        
        image = image[..., 1]
        image = image[..., np.newaxis]
        image = np.concatenate((image,image,image),axis = 2)

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        return image_tensor, mask_tensor



class BrainSegmentationvalDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=10,
        seed=42,
    ):
        assert subset in ["all", "train", "validation"]

        # read images
        self.images = []
        self.masks = []
        self.image_size = 256
        print("reading {} images...".format(subset))
        images_dir = images_dir+'/test'
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                filter(lambda f: ".tif" in f, filenames),
                key=lambda x: int(x.split(".")[-2].split("_")[4]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                     self.masks.append(filepath)
                else:
                     self.images.append(filepath)

        self.random_sampling = random_sampling
        self.transform = transform


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(imread(self.images[idx]))
        mask = np.array(imread(self.masks[idx],as_gray=True))



        image = image[np.newaxis,...]
        mask = mask[np.newaxis,...]

        volumes = [image, mask]
        image, mask = resize_sample(volumes, size = 256)

        image = normalize_volume(image)
        mask = mask[..., np.newaxis]

        image = np.squeeze(image, axis= 0)
        mask = np.squeeze(mask, axis= 0)

        image = image[..., 1]
        image = image[..., np.newaxis]
        image = np.concatenate((image,image,image),axis = 2)

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        return image_tensor, mask_tensor



