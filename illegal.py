from PIL import Image
from tqdm import trange,tqdm
from rembg import remove

import torchvision.transforms as T
import albumentations as A
import numpy as np
import pandas as pd
import os


def remove_background(df):

    #for train

    rembg_image_path=[]
    ids=[]
    rembg_df = pd.DataFrame(df.iloc[:,2:].values,columns=['A','B','C','D','E','F','G','H','I','J'])

    for i in trange(len(df)):
        img_path = df['img_path'][i]
        id = df['id'][i]


        ids.append('Rembg_'+id)
        file_name,_ = img_path.split('/')[-1].split('.')


        image = Image.open('data'+img_path[1:])
        image = remove(image)
        rembg_path = 'data/val_rembg/'+file_name+'.png'

        rembg_image_path.append('.'+rembg_path[4:])
        image.save(rembg_path,format='png')

    rembg_df['id'] = ids
    rembg_df['img_path'] = rembg_image_path

    df.to_csv('data/val_rembg.csv',index=False)

    return df


if __name__ =='__main__':

    df = pd.read_csv('data/train.csv')
    train_df = pd.read_csv('data/train_aug.csv')

    total_images = os.listdir('data/train')
    train_images = list(set([img_path.split('/')[-1] for img_path in train_df['img_path']]))
    val = list(set(total_images) - set(train_images))

    val_index = [ int(val_file.split('.')[0].split('_')[1]) for val_file in val]

    val_df = df.iloc[val_index].reset_index(drop=True)
    # need ""output = output.convert("RGB")"" to train
    #remove_background(test_df,train=False)
    remove_background(val_df)