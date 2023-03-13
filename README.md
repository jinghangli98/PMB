# PMB 🧠 👨‍💻
> Automatic alignment and pathology image reconstruction. The alignment process leverages rembg and pretrained LoFTR and demonstrated the robust out of domain adaptation of transformer models for image correspondances. 


### [Alignment Process](https://github.com/jinghangli98/PMB/blob/main/PMB_alignmentProcess.pdf)

#### Step 1: [preprocessing.sh](https://github.com/jinghangli98/PMB/blob/main/preprocessing.sh)
```bash
bash preprocessing.sh
```
Run this code to transfer the nii images from the storinator to local desktop
#### Step 2: [alignment.py](https://github.com/jinghangli98/PMB/blob/main/alignment.py)
```bash
python3 alignment.py
```
Input the rotation angle and slice locations to locate the matching slices. At the end of the script a ppt file and a reconstructed nii file will be made. 

