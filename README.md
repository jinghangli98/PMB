# PMB
> Automatic alignment and pathology image reconstruction


### [Alignment Process](https://github.com/jinghangli98/PMB/blob/main/PMB_alignmentProcess.pdf)

#### Step 1: [preprocessing.sh](https://github.com/jinghangli98/PMB/blob/main/preprocessing.sh)
Run this code to transfer the nii images from the storinator to local desktop
#### Step 2: [alignment.py](https://github.com/jinghangli98/PMB/blob/main/alignment.py)
Input the rotation angle and slice locations to locate the matching slices
#### Step 3: [alignment.py](https://github.com/jinghangli98/PMB/blob/main/makePPT.py)
Run this code to automatically arrange PPT slides
#### Step 4: [png2nii_cam.py](https://github.com/jinghangli98/PMB/blob/main/png2nii_cam.py)
Run this code to reconstruct nii images from gray scale pathology post mortem images 
