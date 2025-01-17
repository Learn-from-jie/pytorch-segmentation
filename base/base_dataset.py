import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage

class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, base_size=None, augment=True, val=False,
                crop_size=321, scale=True, flip=True, rotate=False, blur=False, return_id=False):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image,image_1,image_2,image_3,image_4, label):
        # if self.crop_size:
        #     h, w = label.shape
        #     # Scale the smaller side to crop size
        #     if h < w:
        #         h, w = (self.crop_size, int(self.crop_size * w / h))
        #     else:
        #         h, w = (int(self.crop_size * h / w), self.crop_size)

        #     image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        #     image_1 = [cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR)  for i in image_1]
        #     image_2 = [cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR)  for i in image_2]
        #     image_3 = [cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR)  for i in image_3]
        #     image_4 = [cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR)  for i in image_4]
        #     label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        #     label = np.asarray(label, dtype=np.int32)

        #     # Center Crop
        #     h, w = label.shape
        #     start_h = (h - self.crop_size )// 2
        #     start_w = (w - self.crop_size )// 2
        #     end_h = start_h + self.crop_size
        #     end_w = start_w + self.crop_size
        #     image = image[start_h:end_h, start_w:end_w]
        #     image_1 = [i[start_h:end_h, start_w:end_w]  for i in image_1]
        #     image_2 = [i[start_h:end_h, start_w:end_w]  for i in image_2]
        #     image_3 = [i[start_h:end_h, start_w:end_w]  for i in image_3]
        #     image_4 = [i[start_h:end_h, start_w:end_w]  for i in image_4]
        #     label = label[start_h:end_h, start_w:end_w]
        return image, image_1,image_2,image_3,image_4, label

    def _augmentation(self, image,image_1,image_2,image_3,image_4, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller 
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        # if self.base_size:
        #     if self.scale:
        #         longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        #     else:
        #         longside = self.base_size
        #     h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
        #     image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        #     image_1 = [cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR) for i in image_1]
        #     image_2 = [cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR) for i in image_2]
        #     image_3 = [cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR) for i in image_3]
        #     image_4 = [cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR) for i in image_4]
        #     label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.scale:
            scales = random.choice([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
            new_h, new_w = int(h * scales), int(w * scales)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            image_1 = [cv2.resize(i, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for i in image_1]
            image_2 = [cv2.resize(i, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for i in image_2]
            image_3 = [cv2.resize(i, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for i in image_3]
            image_4 = [cv2.resize(i, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for i in image_4]
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  
        h, w, _ = image.shape
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT)
            image_1 = [cv2.warpAffine(i, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)     for i in image_1]
            image_2 = [cv2.warpAffine(i, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)     for i in image_2]
            image_3 = [cv2.warpAffine(i, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)     for i in image_3]
            image_4 = [cv2.warpAffine(i, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)     for i in image_4]
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                image_1 = [cv2.copyMakeBorder(i, value=0, **pad_kwargs)     for i in image_1]
                image_2 = [cv2.copyMakeBorder(i, value=0, **pad_kwargs)     for i in image_2]
                image_3 = [cv2.copyMakeBorder(i, value=0, **pad_kwargs)     for i in image_3]
                image_4 = [cv2.copyMakeBorder(i, value=0, **pad_kwargs)     for i in image_4]
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
            # Cropping 
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            image_1 = [i[start_h:end_h, start_w:end_w]  for i in image_1]
            image_2 = [i[start_h:end_h, start_w:end_w]  for i in image_2]
            image_3 = [i[start_h:end_h, start_w:end_w]  for i in image_3]
            image_4 = [i[start_h:end_h, start_w:end_w]  for i in image_4]
            label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                image_1 = [np.fliplr(i).copy() for i in image_1]
                image_2 = [np.fliplr(i).copy()  for i in image_2]
                image_3 = [np.fliplr(i).copy()  for i in image_3]
                image_4 = [np.fliplr(i).copy()  for i in image_4]
                label = np.fliplr(label).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return image, image_1,image_2,image_3,image_4, label
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        center_image = image[0][0]
        image_1  =  image[1]
        image_2 = image[2]
        image_3 = image[3]
        image_4 = image[4]

    # 自定义图片数组，数据类型一定要转为‘uint8’,不然transforms.ToTensor()不会归一化     
        if self.val:
            center_image, image_1,image_2,image_3,image_4,label = self._val_augmentation(center_image,image_1,image_2,image_3,image_4, label)
        elif self.augment:
            center_image, image_1,image_2,image_3,image_4,label = self._augmentation(center_image,image_1,image_2,image_3,image_4, label)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        center_image = Image.fromarray(np.uint8(center_image))
        image_1 = [self.normalize(self.to_tensor(Image.fromarray(np.uint8(i))))  for i in  image_1]
        image_2 = [self.normalize(self.to_tensor(Image.fromarray(np.uint8(i))))  for i in  image_2]
        image_3 = [self.normalize(self.to_tensor(Image.fromarray(np.uint8(i))))  for i in  image_3]
        image_4 = [self.normalize(self.to_tensor(Image.fromarray(np.uint8(i))))  for i in  image_4]
        if self.return_id:
            return  self.normalize(self.to_tensor(center_image)),image_1,image_2,image_3,image_4,label, image_id
        return self.normalize(self.to_tensor(center_image)),image_1,image_2,image_3,image_4 ,label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

