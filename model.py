import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from transformers import  ConvNextModel

class BaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__()
        
        self.model = ConvNextModel.from_pretrained("facebook/convnext-base-224")
        #self.backbone = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.classifier = nn.Linear(1024, 512)
        self.classifier2 = nn.Linear(512,num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.model(**x)
        #x = self.backbone(x)
        x = x.pooler_output
        x = self.relu(self.classifier(x))
        x = F.sigmoid(self.classifier2(x))
        return x