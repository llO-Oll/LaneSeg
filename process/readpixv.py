from PIL import Image
import numpy as np
from cv2 import cv2
import os
from tqdm import tqdm
def pixv(imgpath):
        count=[0]*15
        for filename in os.listdir(imgpath):
                img = np.array(Image.open(imgpath+'/'+filename))
                count[0]+=1
                for i in range(0,15):
                        if img[img==i].any()==True :     
                                count[i]+=1
                                print(count)

def count_pixv(imgpath):
        count=0
        for filename in os.listdir(imgpath):
                img = np.array(Image.open(imgpath+'/'+filename))
                if img[img==1].any()==True :     
                        print(filename)
                        count+=1
                        print(count)

def fusionImageMask(imgpath):
        for filename in os.listdir(imgpath):
                print(filename)
                img = np.array(Image.open(imgpath+'/'+filename))
                print(img[np.where(img==1)])
count_pixv('/home/whut/PaddleSeg/data_one/data_8/mask_8')
# pixv('/home/whut/PaddleSeg/ocrnet_output7/result_tta')

# pixv('/home/whut/PaddleSeg/seocrnet_output13/result_tta')