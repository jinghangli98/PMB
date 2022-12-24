import nibabel as nib
import numpy as np
import pdb
import sys
from rembg import remove
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform
import torchvision.transforms as transforms
from torchvision.transforms.functional import adjust_contrast
from similarities.imagesim import ImageHashSimilarity, SiftSimilarity, ClipSimilarity
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import skimage.util
import os
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *

import glob
import cv2
import matplotlib.pyplot as plt
from natsort import natsorted

def load_torch_image(fname):
    img = remove(cv2.imread(fname))
    img = img[:, :, 0:3]
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img


def get_rotation_matrix(x=0, y=0, z=0):
    """ Computes the rotation matrix.
    
    Parameters
    ----------
    x : float
        Rotation in the x (first) dimension in degrees
    y : float
        Rotation in the y (second) dimension in degrees
    z : float
        Rotation in the z (third) dimension in degrees
    
    Returns
    -------
    rot_mat : numpy ndarray
        Numpy array of shape 4 x 4
    """
    
    x = np.deg2rad(x)
    y = np.deg2rad(y)
    z = np.deg2rad(z)
    
    rot_roll = np.array([
        [1, 0, 0, 0],
        [0, np.cos(x), -np.sin(x), 0],
        [0, np.sin(x), np.cos(x), 0],
        [0, 0, 0, 1]
    ])

    rot_pitch = np.array([
        [np.cos(y), 0, np.sin(y), 0],
        [0, 1, 0, 0],
        [-np.sin(y), 0, np.cos(y), 0],
        [0, 0, 0, 1]
    ])

    rot_yaw = np.array([
        [np.cos(z), -np.sin(z), 0, 0],
        [np.sin(z), np.cos(z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    rot_mat = rot_roll @ rot_pitch @ rot_yaw
    return rot_mat

def get_translation_matrix(x=0, y=0, z=0):
    """ Computes the translation matrix.
    
    Parameters
    ----------
    x : float
        Translation in the x (first) dimension in voxels
    y : float
        Rotation in the y (second) dimension in voxels
    z : float
        Rotation in the z (third) dimension in voxels
    
    Returns
    -------
    trans_mat : numpy ndarray
        Numpy array of shape 4 x 4
    """
    
    trans_mat = np.eye(4)
    trans_mat[:, -1] = [x, y, z, 1]
    return trans_mat

def resample_image(image, trans_mat, rot_mat):
    """ Resamples an image given a translation and rotation matrix.
    
    Parameters
    ----------
    image : numpy array
        A 3D numpy array with image data
    trans_mat : numpy array
        A numpy array of shape 4 x 4
    rot_mat : numpy array
        A numpy array of shape 4 x 4
    
    Returns
    -------
    image_reg : numpy array
        A transformed 3D numpy array
    """
    
    # We need to rotate around the origin, not (0, 0), so
    # add a "center" translation
    center = np.eye(4)
    center[:3, -1] = np.array(image.shape) // 2 - 0.5
    A = center @ trans_mat @ rot_mat @ np.linalg.inv(center)
    
    # affine_transform does "pull" resampling by default, so
    # we need the inverse of A
    image_corr = affine_transform(image, matrix=np.linalg.inv(A))
    
    return image_corr

def zeroCrop(cor_slice):
    col = np.argwhere(abs(np.sum(cor_slice, 0)) <= 100)
    row = np.argwhere(abs(np.sum(cor_slice, 1)) <= 100)
    
    cropped = np.delete(cor_slice, col, 1)
    cropped = np.delete(cropped, row, 0)

    return cropped

def adjustContrast(img):
    

    minval = np.percentile(img, 1)
    maxval = np.percentile(img, 99.5)
    pixvals = np.clip(img, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval)) * 255
    # pixvals.astype(np.uint8)

    plt.subplot(1,2,1)
    plt.imshow(pixvals[:,:,20], 'gray')
    plt.subplot(1,2,2)
    plt.imshow(img[:,:,20], 'gray')
    plt.show()
    pdb.set_trace()

    LUT = np.zeros(256, dtype=np.float64)
    LUT[min:max+1]=np.linspace(start=0.0, stop=255.0, num=(max-min)+1, endpoint=True, dtype=np.float64)

    return LUT[img]


def customResize(move, size):
    target_h, target_w = size
    ratio = move.shape[0]/move.shape[1]
    resized = cv2.resize(move, [ target_w, int(target_w*ratio)], interpolation = cv2.INTER_AREA)
    transform = transforms.CenterCrop((target_h, target_w))
    resized = Image.fromarray(resized)
    resized = transform(resized)
    resized = np.array(resized)

    try:
        assert((resized.shape[0], resized.shape[1]) == (target_h, target_w))
    except:
        pdb.set_trace()
    
    return resized