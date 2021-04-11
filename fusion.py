from PIL import Image
import numpy as np
from cv2 import cv2
import os
from tqdm import tqdm

def image128(imgpath,savepath):
    for filename in tqdm(os.listdir(imgpath)):
        # print(filename)
        img = np.array(Image.open(imgpath+'/'+filename))
        img[img!=8]=0
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            cv2.imwrite(savepath+'/'+filename,img)
        else:
            cv2.imwrite(savepath+'/'+filename,img)

def imageFusion(img1path,img15path,savepath):
    for filename in tqdm(os.listdir(img1path)):
        img1 = np.array(Image.open(img1path+'/'+filename))
        img15 = np.array(Image.open(img15path+'/'+filename))
        img15[np.where(img1==1)]=12
        cv2.imwrite(savepath+'/'+filename,img15)


def imageFusion_pixel(img1path,img15path,savepath):
    for filename in os.listdir(img1path):
        img1 = np.array(Image.open(img1path+'/'+filename))
        img15 = np.array(Image.open(img15path+'/'+filename))
        rows,cols=img15.shape
        for i in range(rows):
            for j in range(cols):
                if img15[i,j]==0 and img1[i,j]==6:
                    img15[i,j]=6
        print(filename)
        cv2.imwrite(savepath+'/'+filename,img15)


# imageFusion('/home/whut/PaddleSeg/fusion_model/re12'  , '/home/whut/PaddleSeg/fusion_model/fusionzzl_8_13_tta', '/home/whut/PaddleSeg/fusion_model/temp')
image128('/home/whut/PaddleSeg/fusion_model/predict+0814+10','/home/whut/PaddleSeg/fusion_model/submit_8')

