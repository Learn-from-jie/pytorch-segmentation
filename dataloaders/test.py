import os
from glob import glob
from PIL import Image
import numpy as np
import fnmatch
import re

split ='train'
root = '/DATA/DATA2/lj/UrbanLF/semantic_segmentation/UrbanLF-Real'
print(os.path.join(root, split))
files=[]
if split in ['train','val','test']: #
    image_dir = os.path.join(root,  split)
    label_dir = os.path.join(root, split)
for img_class in os.listdir(image_dir):
    img_all =[]
    img_mulu = os.path.join(image_dir,img_class)
    if(os.path.isdir(img_mulu)):
        img_all = glob(img_mulu+'/*.png')
        img_all.remove(img_mulu+'/label.png')
        oneimg_dict={}
        oneimg_dict["image"] = img_all
        oneimg_dict["label"] = glob(img_mulu+'/*.npy')[0]
        files.append(oneimg_dict)

img_all=[[],[],[],[],[]]
data_dict = files[10]
img_list = data_dict["image"]
label_path = data_dict["label"]
kk = []
yy=[]
for img_name in img_list:
    img_newname  =  img_name.split('/')[-1]
    if fnmatch.fnmatch(img_newname,"?_5.png"):
        img_all[1].append(np.asarray(Image.open(img_name).convert('RGB'), dtype=np.float32))
    if fnmatch.fnmatch(img_newname,"5_?.png"):
        img_all[2].append(np.asarray(Image.open(img_name).convert('RGB'), dtype=np.float32))
    if(int(img_newname.split(".")[0][0])==int(img_newname.split(".")[0][-1])):
        img_all[3].append(np.asarray(Image.open(img_name).convert('RGB'), dtype=np.float32))
    if(int(img_newname.split(".")[0][0])+int(img_newname.split(".")[0][-1])==10):
        img_all[4].append(np.asarray(Image.open(img_name).convert('RGB'), dtype=np.float32))
    if(img_newname=="5_5.png"):
        img_all[0].append(np.asarray(Image.open(img_name).convert('RGB'), dtype=np.float32))
label = np.load(label_path)
img_id =  int(re.findall('\d+',img_name.split("/")[-2])[0])
