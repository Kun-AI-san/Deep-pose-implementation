"""
author: Karthik Vemireddy
UID: u7348473
"""

import scipy.io
import numpy as np
import os
import cv2 as cv2
import matplotlib.pyplot as plt

diction = {}
mat = scipy.io.loadmat('examples.mat')
def apply_crop(train_set, train_labels):
    '''
    Applies cropping bound to the human subjects and ignores mislabled data.
    Input parameters:
        Train_set: Training image data
        Train_labels: Training labels
    output:
        Train_set_1: Augmented training image data
        Train_labels_1: Augmented Training labels
    '''
    train_set_1 = []
    train_labels_1 = []
    for i, example in enumerate(train_set):
        f=0
        for label in train_labels[i]:
            if label[0]<=5 or label[1]<=5 or label[0]>=example.shape[1]-5 or label[1]>=example.shape[0]-5:
                f=1
                break
        if f==1:
            continue
        train_set_1.append(example)
        train_labels_1.append(train_labels[i])
    return train_set_1, np.array(train_labels_1)

def apply_translate(train_set, train_labels):
    '''
    Applies random transaltion to the images.
    Input Parameters: 
        train_set: Training image data
        Train_labels: Training labels
    Output Parameters:
        new_train_set: Augmented training image data
        new_train_labels: Augmented training labels
    '''
    num_range = 5
    new_data_set = train_set.copy()
    new_train_labels = train_labels.copy()
    for i, example in enumerate(train_set):
        will_translate = np.random.randint(0, 2)
        x = np.random.randint(-num_range, num_range)
        y = np.random.randint(-num_range, num_range)
        if will_translate == 1:
            new_image = np.zeros_like(example)
            if x>0:
                if y>0:
                    new_image[y:, x:] = example[:example.shape[0]-y, :example.shape[1]-x]
                else:
                    new_image[:example.shape[0]+y, x:] = example[-y:, :example.shape[1]-x]
            else:
                if y>0:
                    new_image[y:, :example.shape[1]+x] = example[:example.shape[0]-y, -x:]
                else:
                    new_image[:example.shape[0]+y, :example.shape[1]+x] = example[-y:, -x:]
            new_joints = train_labels[i] + np.array([x,y])          
            new_data_set.append(new_image)
            
            new_train_labels = np.append(new_train_labels, np.array([new_joints]), 0)
    return new_data_set, new_train_labels
            

def apply_flip(train_set, train_labels, symmetric_joints):
    '''
    Applies random flipping to the images.
    Input Parameters: 
        train_set: Training image data
        Train_labels: Training labels
        symmetric_joints: used to correctly annotate flipped labels
    Output Parameters:
        new_train_set: Augmented training image data
        new_train_labels: Augmented training labels
    '''
    new_data_set = train_set.copy()
    new_train_labels = train_labels.copy()
    for i, example in enumerate(train_set):
        will_flip = np.random.randint(0, 2)
        if will_flip == 1:
            new_image = cv2.flip(example,1)
            new_joints = train_labels[i].copy()
            new_joints[:, 0] = (example.shape[1]-1)-train_labels[i,:,0] 
            for i, j in symmetric_joints:
                new_joints[i], new_joints[j] = new_joints[j].copy(), new_joints[i].copy()
            new_data_set.append(new_image)
            new_train_labels = np.append(new_train_labels, np.array([new_joints]), 0)
    return new_data_set, new_train_labels
                

def image_resize(data_set, joints):
    '''
    Applies resizing to the images and scales labels accordingly.
    Input Parameters: 
        data_set: Training/test image data
        joints: Training/test labels
    Output Parameters:
        new_data_set: Augmented training/test image data
        new_joints: Augmented training/test labels
    '''
    new_data_set = np.zeros((len(data_set),227,227,3), dtype = np.uint8)
    new_joints = joints.copy()
    for i, example in enumerate(data_set):
        fx = 227/example.shape[1]
        fy = 227/example.shape[0]
        new_image = cv2.resize(example, (227, 227))
        new_data_set[i] = new_image
        new_joints[i,0::2] = new_joints[i,0::2]*fx
        new_joints[i,1::2] = new_joints[i,1::2]*fy
    return new_data_set, new_joints

#Load labels for the LSP dataset
lsp_mat = scipy.io.loadmat('joints_1.mat')['joints'].T
lsp_mat_2 = scipy.io.loadmat('joints_2.mat')['joints'].transpose(2, 0, 1)

instances_1 = []
instances_2 = []

# Load in the images

for filepath in os.listdir('images_1/'):
    instances_1.append(cv2.imread('images_1/{0}'.format(filepath),1))

for filepath in os.listdir('images_2/'):
    instances_2.append(cv2.imread('images_2/{0}'.format(filepath),1))

#Split data set into train and test sets
train_x = instances_1[0:1000]

test_x = instances_1[1000:2000]

train_labels = lsp_mat[0:1000,:,:2]
#Apply cropping to training set
train_x, train_labels = apply_crop(train_x, train_labels)

#Apply translation to training set
train_x, train_labels = apply_translate(train_x, train_labels)
symmetric_joints = [[8, 9], [7, 10], [6, 11], [2, 3], [1, 4], [0, 5]]

#Apply flipping to training set
train_x, train_labels = apply_flip(train_x, train_labels, symmetric_joints)

instances_2, lsp_mat_2 = apply_crop(instances_2, lsp_mat_2[:,:,:2])
instances_2, lsp_mat_2 = apply_translate(instances_2, lsp_mat_2[:,:,:2])
instances_2, lsp_mat_2 = apply_flip(instances_2, lsp_mat_2, symmetric_joints)
train_x = np.append(train_x, instances_2, 0)
train_labels = np.append(train_labels, lsp_mat_2, 0).reshape(len(train_labels)+len(lsp_mat_2), 28)

test_labels = lsp_mat[1000:2000,:,:2].reshape(1000, 28)

#resize and scale the train/test images and labels
train_x, train_labels = image_resize(train_x, train_labels)
test_x, test_labels = image_resize(test_x, test_labels)

#Save the training and testing datasets
np.save("train_images",train_x)
np.save("train_labels", train_labels)
np.save("test_images", test_x)
np.save("test_labels", test_labels)