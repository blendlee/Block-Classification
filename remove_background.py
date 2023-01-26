from PIL import Image
from tqdm import trange,tqdm
from rembg import remove

import torchvision.transforms as T
import albumentations as A
import numpy as np
import pandas as pd



def remove_background(df,train):

    #for train

    rembg_image_path=[]
    ids=[]
    if train:
        rembg_df = pd.DataFrame(df.iloc[:,2:].values,columns=['A','B','C','D','E','F','G','H','I','J'])
    else:
        rembg_df = pd.DataFrame()



    for i in trange(len(df)):
        img_path = df['img_path'][i]
        id = df['id'][i]


        ids.append('Rembg_'+id)
        file_name,_ = img_path.split('/')[-1].split('.')


        image = Image.open('data'+img_path[1:])
        image = remove(image)
        if train:
            rembg_path = 'data/train_rembg/'+file_name+'.png'
        else:
            rembg_path = 'data/test_rembg/'+file_name+'.png'
        rembg_image_path.append('.'+rembg_path[4:])
        image.save(rembg_path,format='png')

    rembg_df['id'] = ids
    rembg_df['img_path'] = rembg_image_path

    if train:
        df.to_csv('data/train_rembg.csv',index=False)
    else:
        df.to_csv('data/test_rembg.csv',index=False)
    return df

if __name__ =='__main__':

    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    # need ""output = output.convert("RGB")"" to train
    #remove_background(test_df,train=False)
    remove_background(train_df,train=True)

