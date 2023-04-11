from base import BaseDataSet1,BaseDataLoader1
from PIL import Image
from glob import glob
import numpy as np
import scipy.io as sio
from utils import palette
import torch
import os
import cv2
import fnmatch
import re 
class UrbanLFdataset(BaseDataSet1):
    def __init__(self,**kwargs):
        self.num_classes =  14  #多少类
        super(UrbanLFdataset,self).__init__(**kwargs)

    def _set_files(self):
        self.files = []
        if self.split in ['train','val','test']: #
            self.image_dir = os.path.join(self.root,  self.split)
            self.label_dir = os.path.join(self.root, self.split)
        for img_class in os.listdir(self.image_dir):
            img_all =[]
            class_name = os.path.join(self.image_dir,img_class)
            if(os.path.isdir(class_name)):
                img_all = glob(class_name+'/*.png')
                img_all.remove(class_name+'/label.png')
                oneimg_dict={}
                oneimg_dict["image"] = img_all
                oneimg_dict["label"] = glob(class_name+'/*.npy')[0]
                self.files.append(oneimg_dict)
    
    def _load_data(self, index):
        img_all=[[]]
        data_dict = self.files[index]
        img_list = data_dict["image"]
        label_path = data_dict["label"]
        for img_name in img_list:
            img_newname  =  img_name.split('/')[-1]
            if(img_newname=="5_5.png"):
                img_all[0].append(np.asarray(Image.open(img_name).convert('RGB'), dtype=np.float32))
        label = np.load(label_path)
        img_id = int(re.findall('\d+',img_name.split("/")[-2])[0])
        return img_all, label,img_id

        
class UrbanLF1(BaseDataLoader1):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):
            
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]  
        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,  # 这行要不要需要斟酌
            'val': val
        }
        if split in ['train','val','test']: 
            self.dataset = UrbanLFdataset(**kwargs)
        super(UrbanLF1,self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
        


