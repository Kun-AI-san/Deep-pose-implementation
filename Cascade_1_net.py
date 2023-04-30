"""
author: Karthik Vemireddy
UID: u7348473
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, alexnet

class Alex_net_c1(nn.Module):
    '''
    Use an Alexnet network built from scratch as presented in the DeepPose paper. 
    The linear layers 6 and 7 have a dropout of 0.6
    '''
    def __init__(self):
        super(Alex_net_c1, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride = 4, padding=1)
        self.conv2 = nn.Conv2d(96, 256, 5, stride = 1, padding = 2)
        self.conv3 = nn.Conv2d(256, 384, 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(384, 384, 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(384, 256, 3, stride = 1, padding = 1)
        self.fc6 = nn.Linear(9216, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2)
        
    def forward(self, x):
        '''Forward propogation of the pipeline'''
        out = F.relu(self.conv1(x))
        out = F.local_response_norm(out,2)
        out = F.max_pool2d(out, 3, stride = 2)

        out = F.relu(self.conv2(out))
        out = F.local_response_norm(out, 2)
        out = F.max_pool2d(out, 3, stride=2)

        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.max_pool2d(out, 3, stride=2)
        out = out.reshape(out.size(0), -1)
        out = F.dropout(F.relu(self.fc6(out)), p=0.6, training=True)
        out = F.dropout(F.relu(self.fc7(out)), p=0.6, training=True)
        
        return self.fc8(out)

'''References: 
    [1] Toshev, A. and Szegedy, C., 2014. Deeppose: Human pose estimation via deep neural networks. 
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1653-1660).
    https://arxiv.org/abs/1312.4659
'''