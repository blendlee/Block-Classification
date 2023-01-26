from dataset import CustomDataset
from model import BaseModel
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2

import albumentations as A
import pandas as pd
import torch



def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = {'pixel_values' : imgs['pixel_values'].squeeze(1).to(device) }
            #imgs = imgs.float().to(device)
            
            probs = model(imgs)

            probs  = probs.cpu().detach().numpy()
            preds = probs > 0.5
            preds = preds.astype(int)
            predictions += preds.tolist()
    return predictions


if __name__ == '__main__':
    CFG = {
    'IMG_SIZE':400,
    'EPOCHS':5,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':41
    }

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test = pd.read_csv('data/test_rembg.csv')


    test_transform = A.Compose([
                                A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_dataset = CustomDataset(test['img_path'].values, None, transforms = None)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = BaseModel()
    model.load_state_dict(torch.load('models/3_best_model.pth'))

    preds = inference(model, test_loader, device)

    submit = pd.read_csv('data/sample_submission.csv')
    submit.iloc[:,1:] = preds
    submit.to_csv('submit.csv', index=False)
