"""
author: Karthik Vemireddy
UID: u7348473
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class Resnet_stage_1(nn.Module):
    def __init__(self, joint_count):
        '''
        Use a pretrained Resnet34 network for the initial stage which is
        to be finetuned for joint estimation task using 2 extra linear layer 
        '''
        super(Resnet_stage_1, self).__init__()
        self.model = resnet34(pretrained=True)
        self.fc8 = nn.Linear(1000, 4096)
        self.fc9 = nn.Linear(4096,joint_count*2)
        
    def forward(self, x):
        '''Forward propagation for the current network'''
        out = self.model(x)
        out = F.dropout(F.relu(self.fc8(out)), p=0.6, training=True)
        
        return self.fc9(out)