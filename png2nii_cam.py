import nibabel as nib
import numpy as np
from natsort import natsorted
import cv2
import glob
import matplotlib.pyplot as plt
import pdb
import sys

# ID='ADRC_018'
# date='2022.02.15-22.09.41'
# start = 459
# end = 29
ID=sys.argv[1]
date=sys.argv[2]
start=np.int16(sys.argv[3])
end=np.int16(sys.argv[4])


basepath = '/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/03-PMB/PMB_ADRC'
paths = basepath + '/' + date + '/' + ID + '/rembg_cam/resizedCam_adjusted'

paths = glob.glob(f'{paths}/*.png')
paths = natsorted(paths)
# paths.reverse()

imgs = [cv2.imread(path) for path in paths]
imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

nii = basepath + '/' + date + '/' + ID + '/mri/T1.nii.gz'
nii_img = nib.load(nii).get_fdata()
nii_affine = nib.load(nii).affine
nii_array = np.zeros(np.shape(nii_img))
thickness = np.int16(np.ceil((start - end)/len(imgs)))

for i in range(len(imgs)):
    print(f'{start-i*thickness} -> {start-(i+1)*thickness}')
    imgs[i] = np.rot90(imgs[i], 2)
    replicated_img = np.dstack([imgs[i]] * thickness)
    replicated_img = np.transpose(replicated_img, (1,2,0))
    nii_array[:,start-(i+1)*thickness:start-i*thickness,:] = replicated_img

    
nii_png_array = nib.Nifti1Image(nii_array, affine=nii_affine)
out_path = basepath + '/' + date + '/' + ID + '/mri/png.nii.gz'
nib.save(nii_png_array, out_path)