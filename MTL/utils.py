from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from tqdm import trange,tqdm
from rembg import remove

import torchvision.transforms as T
import albumentations as A
import numpy as np
import pandas as pd
import cv2


def split_data(df):

    # for lonely values
    df['label'] = [ str(a)+str(b)+str(c)+str(d)+str(e)+str(f)+str(g)+str(h)+str(i)+str(j) 
                    for a,b,c,d,e,f,g,h,i,j in zip(df['A'],df['B'],df['C'],df['D'],df['E'],df['F'],df['G'],df['H'],df['I'],df['J'])]
    
    s= df['label'].value_counts() ==1
    idx = list(s[s].index)
    for i in idx :
        df = df.append(df.loc[df['label'] == i])

    df= df.reset_index(drop=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for i, (train_index, test_index) in enumerate(sss.split(df, df['label'])):
        train_df = df.loc[train_index]
        val_df = df.loc[test_index]
    
    return train_df,val_df


def augmentation(df):
    transform = A.Compose([A.VerticalFlip(p=1),
                            A.RandomBrightnessContrast(p=0.2),
                            A.Rotate(limit=90, p=0.7, border_mode=cv2.BORDER_REPLICATE)
                                 ])

    aug_image_path = []
    ids=[]
    aug_df = pd.DataFrame(df.iloc[:,2:-1].values,columns=['A','B','C','D','E','F','G','H','I','J'])
    for img_path,id in tqdm(zip(df['img_path'],df['id'])):
        ids.append('Aug_'+id)
        image = Image.open('../data'+img_path[1:])
        file_name = img_path.split('/')[-1]
        image = np.array(image)
        image = transform(image=image)['image']
        pil_image=Image.fromarray(image)
        aug_path = '../data/augmentation/'+file_name
        aug_image_path.append('.'+aug_path[7:])
        pil_image.save(aug_path)

#'./augmentation/train/TRAIN_10108.jpg

    aug_df['id'] = ids
    aug_df['img_path'] = aug_image_path

    df = pd.concat([df,aug_df]).reset_index(drop=True)
    df.to_csv('../data/train_aug.csv',index=False)

    return df