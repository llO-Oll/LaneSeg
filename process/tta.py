import os
from cv2 import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

PRE_RESULT= ['/home/whut/PaddleSeg/fusion_model/zzl_8_13_re12_re14_3.53/',
'/home/whut/PaddleSeg/fusion_model/deeplabv3p_15/',
'/home/whut/PaddleSeg/fusion_model/unet/',
'/home/whut/PaddleSeg/fusion_model/unet1/'
]
savepath='/home/whut/PaddleSeg/fusion_model/result_tta'
def tta_perimage( image_id,savepath):
    result_list=[]
    for i in range(len(PRE_RESULT)):
        image = cv2.imread((PRE_RESULT[i])+str(image_id)+'.png',0)
        result_list.append(image)
        h,w=result_list[0].shape
        vote_mask = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                record = np.zeros((1,15))
                for n in range(len(result_list)):
                    mask = result_list[n]
                    pixel =mask[i,j]
                    record[0,pixel]+=1
                    
                label = record.argmax()
                vote_mask[i,j] = label
    
        cv2.imwrite(savepath+'/'+str(image_id)+'.png',vote_mask)

def tta(savepath):
    for i in tqdm(range(4000,5000)):
        tta_perimage(i,savepath)

tta(savepath)
