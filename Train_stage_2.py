"""
author: Karthik Vemireddy
UID: u7348473
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from Resnet_stage_1 import Resnet_stage_1
from Cascade_1_net import Alex_net_c1
import torch.nn as nn
import cv2 as cv2
from random import shuffle
#Assign CUDA resources if available
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = np.load("./train_images.npy")
y = np.load("./train_labels.npy")
y_1 = np.load("./train_output.npy")
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
#uncomment the following lines of code for creating simulated displacements for augmenting the data (Mandatory for first time run)
#mean = []
#var = []
#np.random.RandomState(421)
#deltas = []
#x = np.zeros((11000, 14, 220, 220, 3), dtype=np.uint8)
joint_list = ["right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle", "right_wrist", "right_elbow", "right_shoulder", "left_shoulder", "left_elbow", "left_wrist", "neck", "head"]

def crop_resize(image_data, labels, predictions, joint):
    '''
    Crops a sub-image from a training sample around an estimated joint from stage 1.
    Each joint gets its own sub-image with normalized labels and predictions.
    This function is also further augments the training data by adding 20 simulated new predictions
    for a joint of an image based on mean and variance of displacement between predicted and ground truth points.
    Input Parameters: 
        image_data: Training images
        labels: Ground truth points for joints on the image
        predictions: Estimated joint points from previous layer of outputs
        joint: Joint being computed currenty
    Output Parameters:
        X: Augmented or cropped and resized images for the joint.
        Y: Augmented and normalized ground tructh points
        l1_out: Augmented and normalized previous layer's estimated points
        coordinates: Scaling factors to preserve the original scale of the points 
    '''
    coordinates = []
    X = torch.zeros((image_data.shape[0]*20, 3, 227, 227))
    Y = torch.zeros((image_data.shape[0]*20, 2))
    l1_out = torch.zeros((image_data.shape[0]*20, 2))
    k=0
    ind = joint_list.index(joint)
    for i, image in enumerate(image_data):
        y = np.zeros(labels[i].shape[0])
        y_t = np.zeros(labels[i].shape[0])
        y1 = torch.zeros(2)
        y_t = labels[i].detach().cpu().numpy()*image.shape[1]+image.shape[1]/2
        y[0::2] = predictions[i, 0::2].detach().cpu().numpy()
        y[1::2] = predictions[i, 1::2].detach().cpu().numpy()
        #Create an arbitrary bounding box based on right hip and left shoulder or their symmetric counterparts
        bbox_dim = 2*np.sqrt((y[6]-y[16])**2 + (y[7]-y[17])**2)
        j = ind*2
        delta = deltas[ind]
        for l in range(20):
            y_sample = delta[l]
            x_low = min(image.shape[2]-1,max(0, y[j]+y_sample[0]-(bbox_dim/2)))
            y_low = min(image.shape[1]-1,max(0, y[j+1]+y_sample[1]-(bbox_dim/2)))
            x_up = max(0, min(image.shape[2]-1, y[j]+y_sample[0]+(bbox_dim/2)))
            y_up = max(0, min(image.shape[1]-1, y[j+1]+y_sample[1]+(bbox_dim/2)))
            if labels[i,j]*227+113.5 < x_low:
                x_low = max(0,(labels[i,j]*227+113.5)-10)
            elif labels[i,j]*227+113.5 > x_up:
                x_up = min(image.shape[2]-1,(labels[i,j]*227+113.5)+10)
            if labels[i,j+1]*227+113.5 < y_low:
                y_low = max(0,(labels[i,j+1]*227+113.5)-10)
            elif labels[i,j+1]*227+113.5 > y_up:
                y_up = min(image.shape[1]-1, (labels[i,j+1]*227+113.5)+10)
            image_1 = image.permute(1,2,0).detach().cpu().numpy()
            y1[0] = (labels[i,j]*227+113.5)-x_low
            y1[1] = (labels[i,j+1]*227+113.5)-y_low
            image_1 = (image_1[int(y_low):int(y_up)+1, int(x_low):int(x_up)+1])
            
            fx = 227/image_1.shape[1]
            fy = 227/image_1.shape[0]
            image_1 = cv2.resize(image_1, (227, 227))
            y1[0] = y1[0]*fx
            y1[1] = y1[1]*fy
            coordinates.append(np.array([x_low, y_low, fx, fy]))
            X[k] = torch.tensor(image_1.transpose(2,0,1))
            Y[k] = (y1-113.5)/227
            y1[0] = (y[j]+y_sample[0]-x_low)*fx
            y1[1] = (y[j+1]+y_sample[1]-y_low)*fy
            l1_out[k] = (y1-113.5)/227
            k+=1
    return X, Y, l1_out, coordinates


class CustomDataSet(Dataset):
    def __init__(self, data_file, data_labels, l1_label_out):
        self.x = np.load(data_file)
        self.y = np.load(data_labels)
        self.l1_output = np.load(l1_label_out)
        a = self.x[0].shape[0]/2
        b = self.x[0].shape[1]/2
        #uncomment this code for creating simulated displacements for augmenting the data (Mandatory for first time run)
        #for i in range(14):
        #    mean.append(np.mean(self.l1_output[:,i*2:i*2+2]-self.y[:,i*2:i*2+2], axis=0))
        #    var.append(np.cov((self.l1_output[:,i*2:i*2+2]-self.y[:,i*2:i*2+2]).T))
        self.y[:, 0::2] = (self.y[:, 0::2]-b)/self.x[0].shape[1]
        self.y[:, 1::2] = (self.y[:, 1::2]-a)/self.x[0].shape[0]
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        
        image = transform(self.x[index])
        label = self.y[index]
        l1_out = torch.tensor(self.l1_output[index])
        return image, label, l1_out
    
train_data = CustomDataSet("./train_images.npy", "./train_labels.npy", "./train_output.npy")

#Create a test data-loader using custom dataset with batch size of 10.
train_dataloader = DataLoader(train_data, batch_size=10, shuffle = True)
#uncomment the following lines of code for creating simulated displacements for augmenting the data (Mandatory for first time run)
#for i,joint in enumerate(joint_list):
#    deltas.append(np.random.multivariate_normal(mean[i],var[i],size=40))
#np.save("deltas.npy",np.array(deltas))

deltas = np.load("./deltas.npy")
#Total epochs: 7*14  joints = 98 Due to time limitations. I would have liked to try a higher value.
epochs = 7
#Stage 2 Train pipeline
for joint in joint_list:
    model = Alex_net_c1()
    model = model.to(dev)
    model.train()
    #MSE Loss
    loss_1 = nn.MSELoss()
    #Adagrad Optimizer
    optimizer_1 = optim.Adagrad(model.parameters(), lr = 5e-4)
    min_loss = 99999
    for i in range(epochs):
        epoch_loss = 0
        j=0
        for batch_data in train_dataloader:
            X, Y, layer1_output = batch_data
            X_joint, Y_joint, l1_out, coordinates = crop_resize(X, Y, layer1_output, joint)
            X_joint = X_joint.float()
            X_joint, Y_joint, l1_out = X_joint.to(dev), Y_joint.to(dev), l1_out.to(dev)
            p = torch.randperm(X_joint.shape[0])
            q = p.numpy().astype('uint8')
            predictions_1 = model(X_joint[p]*255)
            
            optimizer_1.zero_grad()
            j+=1
            Loss = loss_1(predictions_1+l1_out[p], Y_joint[p])
            epoch_loss+=Loss.item()/(train_data.__len__()/10)
            Loss.backward()
            optimizer_1.step()
        print("Epoch ", i+1, " Loss for ", joint,": ", epoch_loss)
        if epoch_loss<min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), "model_"+joint)
