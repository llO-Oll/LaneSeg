import os
from cv2 import cv2
import numpy as np
from PIL import Image
import random
import shutil
from tqdm import tqdm
maskpath='/home/whut/PaddleSeg/mask_4000'
imgpath='/home/whut/PaddleSeg/image_4000'
toMaskFileDir='/home/whut/PaddleSeg/data_one/data_7/n7mask_213negtivate/'
toOriFileDir='/home/whut/PaddleSeg/data_one/data_7/n7ori_213negtivate/'
image_class=7

def get0neClass(maskpath,imgpath,mask_savepath,ori_savepath,image_class):
    print(os.path.exists(mask_savepath) and  os.path.exists(mask_savepath))
    for filename in os.listdir(maskpath):
        mask = np.array(Image.open(maskpath+'/'+filename))
        if mask[mask==image_class].any()==True:
            print(filename)
            mask[mask!=image_class]=0
            mask[mask==image_class]=1
            ori = np.array(Image.open(imgpath+'/'+filename))

            if not os.path.exists(mask_savepath) and  os.path.exists(ori_savepath):
                 os.makedirs(mask_savepath)
                 os.makedirs(ori_savepath)
                 cv2.imwrite(mask_savepath+'/'+filename,mask)
                 cv2.imwrite(ori_savepath+'/'+filename,mask)
            else:
                cv2.imwrite(mask_savepath+'/'+filename,mask)
                cv2.imwrite(ori_savepath+'/'+filename,ori)

def getBlack(maskpath,mask_savepath):
    for filename in tqdm(os.listdir(maskpath)):
        mask = np.array(Image.open(maskpath+'/'+filename))
        mask[mask!=0]=0
        if not os.path.exists(mask_savepath):
            os.makedirs(mask_savepath)        
            cv2.imwrite(mask_savepath+'/'+filename,mask)
        else:
            cv2.imwrite(mask_savepath+'/'+filename,mask)            
    
def getNegSameple(mask_savepath,ori_savepath,image_class):
    for filename in tqdm(os.listdir(maskpath)):
        mask=np.array(Image.open(maskpath+'/'+filename))
        if mask[mask==image_class].any()==False:
            if not (os.path.exists(mask_savepath) and  os.path.exists(ori_savepath)):
                os.makedirs(mask_savepath)
                os.makedirs(ori_savepath)
                cv2.imwrite(mask_savepath+'/'+filename,mask)
                ori=np.array(Image.open(imgpath+'/'+filename))
                cv2.imwrite(ori_savepath+'/'+filename,ori)
            else:
                cv2.imwrite(mask_savepath+'/'+filename,mask)
                ori=np.array(Image.open(imgpath+'/'+filename))
                cv2.imwrite(ori_savepath+'/'+filename,ori)


def getPng(filename:str):
    return filename.endswith("png")

def getFile(maskFileDir, toMaskFileDir, oriFileDir,toOriFileDir,number):
    mask_path = os.path.abspath(maskFileDir) + "/"
    ori_path = os.path.abspath(oriFileDir) + "/"
    if os.path.isdir(maskFileDir):
        pathList = os.listdir(maskFileDir)
        pngList = [i for i in filter(getPng, pathList)]
        if pngList:
            sample = random.sample(pngList, number)
            for name in sample:
                if not (os.path.exists(toMaskFileDir) and  os.path.exists(toOriFileDir)):
                    os.makedirs(toMaskFileDir)
                    os.makedirs(toOriFileDir)                
                    shutil.copy(mask_path+name, toMaskFileDir)
                    shutil.copy(ori_path+name, toOriFileDir)
                else:
                    shutil.copy(mask_path+name, toMaskFileDir)
                    shutil.copy(ori_path+name, toOriFileDir)



get0neClass(maskpath,imgpath,'/home/whut/PaddleSeg/data_one/data_7/mask_7','/home/whut/PaddleSeg/data_one/data_7/ori_7',7)
# getNegSameple('/home/whut/PaddleSeg/data_one/data_7/n7mask','/home/whut/PaddleSeg/data_one/data_7/n7ori',image_class)
# getFile('/home/whut/PaddleSeg/data_one/data_7/n7mask',toMaskFileDir,'/home/whut/PaddleSeg/data_one/data_7/n7ori' , toOriFileDir , 118)
# getBlack('/home/whut/PaddleSeg/data_one/data_7/n7mask_213negtivate','/home/whut/PaddleSeg/data_one/data_7/n7mask_213negtivate1')