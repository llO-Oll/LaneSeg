import numpy as np
from cv2 import cv2
import os
from PIL import Image
from tqdm import tqdm
def gray2rgb(imgpath,savepath):
    for filename in os.listdir(imgpath):
        img = np.array(Image.open(imgpath+'/'+filename))
        if img[img==1].any()==True:
            print(filename)
            rows,cols=img.shape
            img_rgb=np.zeros(shape=(rows,cols,3),dtype=np.uint8)
            img_rgb[img==1]=[255,0,0] 
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                cv2.imwrite(savepath+'/'+filename,img_rgb)
            else:
                cv2.imwrite(savepath+'/'+filename,img_rgb)
def gray2rgb15(imgpath,savepath):
    for filename in tqdm(os.listdir(imgpath)):
        img = np.array(Image.open(imgpath+'/'+filename))
        rows,cols=img.shape
        img_rgb=np.zeros(shape=(rows,cols,3),dtype=np.uint8)
        img_rgb[img==1]=[255,0,0]
        img_rgb[img==2]=[255,255,0]
        img_rgb[img==3]=[150,150,0]
        img_rgb[img==4]=[150,100,0]
        img_rgb[img==5]=[150,200,0]
        img_rgb[img==6]=[0,255,0]
        img_rgb[img==7]=[0,255,255]
        # img_rgb[img==8]=[0,150,150]
        img_rgb[img==9]=[0,150,200]
        img_rgb[img==10]=[0,200,255]
        img_rgb[img==11]=[0,0,255]
        img_rgb[img==12]=[255,0,255]
        img_rgb[img==13]=[110,0,145]
        img_rgb[img==14]=[0,100,0]
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            cv2.imwrite(savepath+'/'+filename,img_rgb)
        else:
            cv2.imwrite(savepath+'/'+filename,img_rgb)           

def overlying(oripath,imgpath,savepath):
    for filename in tqdm(os.listdir(oripath)):
        ori =cv2.imread(oripath+'/'+filename)
        img =cv2.imread(imgpath+'/'+filename)
        res =cv2.addWeighted(ori,0.4,img,0.6,0)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            cv2.imwrite(savepath+'/'+filename,res)
        else:
            cv2.imwrite(savepath+'/'+filename,res)    

def test(imagepath):
    for filename in os.listdir(imagepath):
        img=cv2.imread(imagepath+'/'+filename)
        sp=img.shape
        print(sp)
        

gray2rgb15('/home/whut/PaddleSeg/fusion_model/predict+0814+10' , '/home/whut/PaddleSeg/fusion_model/rgb_n8')
overlying('/home/whut/PaddleSeg/fusion_model/rgb_n8','/home/whut/PaddleSeg/data/infer','/home/whut/PaddleSeg/fusion_model/overlying_rgb_n8')
