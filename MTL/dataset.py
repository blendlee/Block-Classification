
from torch.utils.data import Dataset
from transformers import ConvNextFeatureExtractor
from PIL import Image


import cv2
import torch

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        self.feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-base-224")



    def __getitem__(self, index):
        img_path = '../data' + self.img_path_list[index][1:]
        
        image = Image.open(img_path)
        image = image.convert('RGB')
        image= self.feature_extractor(images=image, return_tensors="pt")
        #image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
            
        
        if self.label_list is not None:
            label = torch.FloatTensor(self.label_list[index])
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)