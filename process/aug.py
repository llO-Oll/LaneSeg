import random
from cv2 import cv2
from matplotlib import pyplot as plt
import albumentations as A
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
maskpath='/home/whut/PaddleSeg/data_one/data_7/mask_7'
imagepath='/home/whut/PaddleSeg/data_one/data_7/ori_7'
masksavepath='/home/whut/PaddleSeg/data_one/data_7/mask_flip_7'
imagesavepath='/home/whut/PaddleSeg/data_one/data_7/ori_flip_7'

# image=cv2.imread('/home/whut/PaddleSeg/image_4000/1.png')
# mask=cv2.imread('/home/whut/PaddleSeg/mask_4000/1.png')



def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
       
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()


# aug = A.Compose([
#     A.OneOf([A.VerticalFlip(p=0.5),      
#     A.HorizontalFlip(p=0.5)
#     ]),

#     A.OneOf([
#            A.RandomRotate90(p=1)
#     ])
# ])
# aug = A.Compose([
#     A.VerticalFlip(p=0.5),              
#     A.HorizontalFlip(p=1)]
#  )
def aug_rot(angle,image,mask):
    aug = A.Compose([
    A.Rotate([angle,angle],interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1)
    ])
    return aug(image=image, mask=mask)
# aug = A.Compose([
#  A.Transpose(p=1)
#  ])



# aug = A.Compose([
#  A.RandomCrop(1024,1024,p=1)
#  ])
angle=[30,60,120,150]
for filename in tqdm(os.listdir(maskpath)):
    img = np.array(Image.open(maskpath+'/'+filename))
    mask = cv2.imread(maskpath+'/'+filename,cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(imagepath+'/'+filename)
    for i in angle:
        augmented = aug_rot( i,image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        if not (os.path.exists(masksavepath) and  os.path.exists(imagesavepath)):
            os.makedirs(masksavepath)
            os.makedirs(imagesavepath)        
            cv2.imwrite(masksavepath+'/'+'rot'+ str(i)+'_'+filename, aug_mask)
            cv2.imwrite(imagesavepath+'/'+'rot'+str(i)+'_'+filename, aug_image)
        else:

            cv2.imwrite(masksavepath+'/'+'rot'+ str(i)+'_'+filename, aug_mask)
            cv2.imwrite(imagesavepath+'/'+'rot'+str(i)+'_'+filename, aug_image)            



# visualize(image_scaled, mask_scaled, original_image=image, original_mask=mask)

    