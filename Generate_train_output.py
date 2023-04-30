# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:46:34 2022

@author: Karthik
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

train_dataloader = DataLoader(train_data, batch_size=64, shuffle = False)

model = Resnet_stage_1(14)
model.load_state_dict(torch.load("model_1"))
model.to(dev)
optimizer = optim.Adam(model.parameters(), lr = 5e-4)
loss = nn.MSELoss()

print(len(train_data))
output = []
model.eval()
x = []
y = []
prediction_2 = []
with torch.no_grad():
    epoch_loss = 0
    for batch_data in train_dataloader:
        X, Y = batch_data
        x_shape = X.shape[0]
        for j, x1 in enumerate(X):
            x.append(np.array(x1.numpy().transpose((1,2,0))*255, dtype=np.uint8))
            y.append(np.array(Y[j]))
        
        X = X.float()
        Y = Y.float()
        X = X.to(dev)
        Y = Y.to(dev)
        predictions = model(X*255)
        Loss = loss(predictions, Y)
        for prediction in predictions:
            prediction_2.append(prediction.detach().cpu().numpy())
    print("Test Loss: ", epoch_loss)
x = np.array(x)
y = np.array(y)
prediction_2 = np.array(prediction_2)*227+113.5

np.save("train_output", np.array(prediction_2))