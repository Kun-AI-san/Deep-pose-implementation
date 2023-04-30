"""
author: Karthik Vemireddy
UID: u7348473
"""

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from Resnet_stage_1 import Resnet_stage_1
import torch.nn as nn
#Assign CUDA resources if available
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pdj(pred, Y, thresh = 0.5):
    '''
    Computes the Percentage of detected joints metric for the predictions against the ground truch limb points (Average).
    Input parameters: 
        y: Ground truth points
        pred: Predicted Points from stage 1
        thresh: Threshold for considering whether a joint is correctly classified or not.
    Output parameters:
        None
    '''
    num_correct_parts = 0
    for i in range(Y.shape[0]):
        dist = thresh*np.sqrt((Y[i, 6]-Y[i, 16])**2 + (Y[i, 7]-Y[i, 17])**2)
        for j in range(int(Y.shape[1]/2)):
            pred_dist = np.linalg.norm(np.array([pred[i,j*2:j*2+2]-Y[i,j*2:j*2+2]]))
            if pred_dist<=dist:
                num_correct_parts+=1
    return num_correct_parts/(Y.shape[0]*14)
        

class CustomDataSet(Dataset):
    def __init__(self, data_file, data_labels):
        self.x = np.load(data_file)
        self.y = np.load(data_labels)
        a = self.x[0].shape[0]/2
        b = self.x[0].shape[1]/2
        print(a,b)
        self.y[:, 0::2] = (self.y[:, 0::2]-b)/self.x[0].shape[1]
        self.y[:, 1::2] = (self.y[:, 1::2]-a)/self.x[0].shape[0]
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        image = transform(self.x[index])
        label = torch.tensor(self.y[index])
        return image, label
    
train_data = CustomDataSet("./train_images.npy", "./train_labels.npy")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_dataloader = DataLoader(train_data, batch_size=64, shuffle = True)
test_data = CustomDataSet("./test_images.npy", "./test_labels.npy")
test_dataloader = DataLoader(test_data, batch_size=64, shuffle = False)

model = Resnet_stage_1(14)
#model.load_state_dict(torch.load("model_1"))
model.to(dev)
optimizer = optim.Adam(model.parameters(), lr = 5e-4)
loss = nn.MSELoss()
#Total epochs: 200
epochs = 200
max_test = 0.52
print(len(train_data))
output = []
#Stage 1 Train pipeline
for i in range(epochs):
    epoch_loss = 0
    model.train()
    j = 0
    pdj_acc = 0
    for batch_data in train_dataloader:
        X, Y = batch_data
        X = X.float()*255
        Y = Y.float()
        X = X.to(dev)
        Y = Y.to(dev)
        
        predictions = model(X)
        optimizer.zero_grad()
        Loss = loss(predictions, Y)
        epoch_loss += Loss.item()/(train_data.__len__()/64)
        #print("Iteration ", j+1, " loss: ", Loss.item())
        Loss.backward()
        optimizer.step()
        j+=1
    print("Epoch ", i+1, " Loss: ", epoch_loss)
    epoch_loss = 0
    model.eval()
    
    for batch_data in test_dataloader:
        X, Y = batch_data
        x = np.array(X[1], dtype=np.uint8)
        print(X.shape, Y.shape)
        y = Y[1]
        X = X.float()*255
        Y = Y.float()
        X = X.to(dev)
        Y = Y.to(dev)
        predictions = model(X)
        Loss = loss(predictions, Y)
        epoch_loss += Loss.item()/(test_data.__len__()/64)
        pdj_acc += pdj(predictions.detach().cpu().numpy(), Y.detach().cpu().numpy())/(test_data.__len__()/64)
    print("Epoch ", i+1, "test pdj: ", pdj_acc, epoch_loss)    
    
    if pdj_acc>max_test:
        max_test = pdj_acc
        torch.save(model.state_dict(), "model_1")


#for i in range(len(output)):
#    output[i] = output[i].detach().cpu().numpy()
#np.save("train_output", np.array(output))