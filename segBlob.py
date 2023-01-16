import numpy as np
import glob
from natsort import natsorted
import cv2
import pdb
import numpy.ma as ma
import matplotlib.pyplot as plt
import cv2
import numpy as np
from rembg import remove
from tqdm import tqdm 
import os
import sys

# ID='ADRC_45'
# date='2022.10.26-22.00.00'
# CW_ID='CW22-31'
# base_path='/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/03-PMB/PMB_ADRC'

base_path=sys.argv[1]
date=sys.argv[2]
ID=sys.argv[3]
CW_ID=sys.argv[4]

# read image
images = glob.glob(f'{base_path}/{date}/{CW_ID}/*anterior*/*')
images = natsorted(images)
images = [cv2.imread(img_p) for img_p in images]

rembg_img = []
for idx in tqdm(range(len(images))):
    input = images[idx]
    rembg_img.append(remove(input))
    
# convert to grayscale
def boxCrop(input_img, count):
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    template = np.zeros_like(gray)

    gray = np.argwhere(gray > 50)
    for i in range(len(gray)):
        row, col = gray[i]
        template[row, col] = 255
        
    # threshold
    thresh = cv2.threshold(template,1,255,cv2.THRESH_BINARY)[1]

    # get contours
    result = input_img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    crop_img = []
    box = []
    box_area = []
    for cntr in contours:

        x,y,w,h = cv2.boundingRect(cntr)
        if w*h > 300000:
            
            # cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            print(f'Area of the bounding box is {w*h}')
            print("x,y,w,h:",x,y,w,h)
            
            box.append([x,y,w,h])
            box_area.append(w*h)                  
    
    box = natsorted(box)
    if len(box) == 1 and box_area[0] > 5000000:
        x,y,w,h = box[0]
        box = []
        box = [[x, y, w//2, h], [x + w//2, y, w//2, h]]
    print(f'{box} | {box_area}')    
    
    for i in range(len(box)):
        x,y,w,h = box[i]
        crop_img.append(result[y:-1, x:x+w])  
    
    # save resulting image
    for i in range(len(crop_img)):
        cv2.imwrite(f'{base_path}/{date}/{CW_ID}/cam/{count}.jpg', crop_img[i]) 
        count = count + 1
        
    return count

try:
    os.mkdir(f'{base_path}/{date}/{CW_ID}/cam')
except:
    pass

count = 0
for idx in tqdm(range(len(rembg_img))):
    count = boxCrop(rembg_img[idx], count)

os.system(f'mv {base_path}/{date}/{CW_ID}/cam {base_path}/{date}/{ID}')

