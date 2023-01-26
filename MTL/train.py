

from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np
import torch
import statistics
def train(model, CFG,optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)

    #criterion = nn.BCELoss().to(device)
    criterion = [nn.CrossEntropyLoss().to(device) for _ in range(10)]
    
    best_val_acc = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            #imgs = imgs.float().to(device)
            #print(imgs)
            imgs = {'pixel_values' : imgs['pixel_values'].squeeze(1).to(device) }
            
            #labels = labels.to(device)
            labels = labels.T.type(torch.LongTensor).to(device)
            
            optimizer.zero_grad()

            #output = model(imgs)
            A_result,B_result,C_result,D_result,E_result,F_result,G_result,H_result,I_result,J_result = model(imgs)

            #loss = criterion(output, labels)
            loss_A = criterion[0](A_result,labels[0])
            loss_B = criterion[1](B_result,labels[1])
            loss_C = criterion[2](C_result,labels[2])
            loss_D = criterion[3](D_result,labels[3])
            loss_E = criterion[4](E_result,labels[4])
            loss_F = criterion[5](F_result,labels[5])
            loss_G = criterion[6](G_result,labels[6])
            loss_H = criterion[7](H_result,labels[7])
            loss_I = criterion[8](I_result,labels[8])
            loss_J = criterion[9](J_result,labels[9])
            loss = (loss_A+loss_B+loss_C+loss_D+loss_E+loss_F+loss_G+loss_H+loss_I+loss_J)/10

            
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        #_val_loss, _val_acc = validation(model, criterion, val_loader, device)
        val_loss, val_acc = validation(model, criterion, val_loader, device)
        _val_loss = statistics.mean(val_loss)
        _val_acc = statistics.mean(val_acc)

        _train_loss = np.mean(train_loss)
        #print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]')
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]')
        l=['A','B','C','D','E','F','G','H','I','J']
        for i in range(10):
            print(f'{l[i]} Label Result || Val Loss : [{val_loss[i]:.5f}] Val ACC : [{val_acc[i]:.5f}]')


        if scheduler is not None:
            scheduler.step(_val_acc)
            
        if best_val_acc < _val_acc:
            best_val_acc = _val_acc
            best_model = model
            torch.save(best_model.state_dict(), f'models/{epoch}_best_model.pth')
        else:
            torch.save(best_model.state_dict(), f'models/{epoch}_model.pth')

    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    #val_loss = []
    val_loss = [[] for _ in range(10)]
    #val_acc = []
    val_acc = [[] for _ in range(10)]

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = {'pixel_values' : imgs['pixel_values'].squeeze(1).to(device) }

            #imgs = imgs.float().to(device)
            #labels = labels.to(device)
            labels = labels.T.type(torch.LongTensor).to(device)
            
            #probs = model(imgs)
            A_result,B_result,C_result,D_result,E_result,F_result,G_result,H_result,I_result,J_result = model(imgs)
            results = [A_result,B_result,C_result,D_result,E_result,F_result,G_result,H_result,I_result,J_result]

            #loss = criterion(probs, labels)
            loss_A = criterion[0](A_result,labels[0])
            loss_B = criterion[1](B_result,labels[1])
            loss_C = criterion[2](C_result,labels[2])
            loss_D = criterion[3](D_result,labels[3])
            loss_E = criterion[4](E_result,labels[4])
            loss_F = criterion[5](F_result,labels[5])
            loss_G = criterion[6](G_result,labels[6])
            loss_H = criterion[7](H_result,labels[7])
            loss_I = criterion[8](I_result,labels[8])
            loss_J = criterion[9](J_result,labels[9])
            loss = (loss_A+loss_B+loss_C+loss_D+loss_E+loss_F+loss_G+loss_H+loss_I+loss_J)/10
            losses=[loss_A,loss_B,loss_C,loss_D,loss_E,loss_F,loss_G,loss_H,loss_I,loss_J]


            #probs  = probs.cpu().detach().numpy()
            for i in range(10):
                result = results[i].cpu().detach().numpy()
                label = labels[i].cpu().detach().numpy()
                pred = np.argmax(result,axis=1)
                acc = (pred == label).mean()
                val_acc[i].append(acc)
                val_loss[i].append(losses[i].item())

            #labels = labels.cpu().detach().numpy()
            #preds = probs > 0.5
            #batch_acc = (labels == preds).mean()
            
            #val_acc.append(batch_acc)
            #val_loss.append(loss.item())
        
        #_val_loss = np.mean(val_loss)
        #_val_acc = np.mean(val_acc)
        for i in range(10):
            val_acc[i] = np.mean(val_acc[i])
            val_loss[i] = np.mean(val_loss[i])
    
    #return _val_loss, _val_acc
    return val_loss, val_acc