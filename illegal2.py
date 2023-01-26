import pandas as pd
import os
from PIL import Image
from tqdm import trange

if __name__ =='__main__':


    df = pd.read_csv('data/val_rembg.csv')

    saved_path=[]
    bgn_path = os.listdir('data/test_bgd')
    bgn_path = [ path for path in bgn_path if 'ipy' not in path]
    cnt=0
    for i in trange(len(df)):
        img_path = df['img_path'][i]
        image = Image.open('data/val_rembg/'+img_path.split('/')[-1].split('.')[0]+'.png')
        file_name = img_path.split('/')[-1].split('.')[0]
        background = Image.open('data/test_bgd/'+bgn_path[cnt%len(bgn_path)])
        background.paste(image,(0,0),image)
        background.save(f'data/valid/'+file_name+'.jpg')
        
        saved_path.append(f'./valid/'+file_name+'.jpg')
        cnt+=1
        
    df['img_path'] = saved_path
    df.to_csv('data/valid.csv')