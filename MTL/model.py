import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from transformers import  ConvNextModel

class BaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__()
        
        self.model = ConvNextModel.from_pretrained("facebook/convnext-base-224")
        
        #self.classifier = Classifier(num_classes)

        self.A_classifier = nn.Linear(1024, num_classes)
        self.B_classifier = nn.Linear(1024, num_classes)
        self.C_classifier = nn.Linear(1024, num_classes)
        self.D_classifier = nn.Linear(1024, num_classes)
        self.E_classifier = nn.Linear(1024, num_classes)
        self.F_classifier = nn.Linear(1024, num_classes)
        self.G_classifier = nn.Linear(1024, num_classes)
        self.H_classifier = nn.Linear(1024, num_classes)
        self.I_classifier = nn.Linear(1024, num_classes)
        self.J_classifier = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(**x)
        x = x.pooler_output

        A_result = self.A_classifier(x)
        #A_result = F.softmax(A_result)

        B_result = self.B_classifier(x)
        #B_result = F.softmax(B_result)

        C_result = self.C_classifier(x)
        #C_result = F.softmax(C_result)

        D_result = self.D_classifier(x)
        #D_result = F.softmax(D_result)

        E_result = self.E_classifier(x)
        #E_result = F.softmax(E_result)

        F_result = self.F_classifier(x)
        #F_result = F.softmax(F_result)

        G_result = self.G_classifier(x)
        #G_result = F.softmax(G_result)

        H_result = self.H_classifier(x)
        #H_result = F.softmax(H_result)

        I_result = self.I_classifier(x)
        #I_result = F.softmax(I_result)

        J_result = self.J_classifier(x)
        #J_result = F.softmax(J_result)


        #x = F.sigmoid(self.classifier2(x))
        #return x

        return [A_result,B_result,C_result,D_result,E_result,F_result,G_result,H_result,I_result,J_result]



class Classifier(nn.Module):

    def __init__(self,num_classes):
        super(Classifier, self).__init__()

        self.classifier1 = nn.Linear(1024,512)
        self.classifier2 = nn.Linear(512,num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.classifier1(x)) 
        x = self.classifier2(x) 

        return x