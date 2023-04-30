"""
author: Karthik Vemireddy
UID: u7348473
"""

import torch
from Resnet_stage_1 import Resnet_stage_1
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from Cascade_1_net import Alex_net_c1
import cv2 as cv2

#Assign CUDA resources if available
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Resnet_stage_1(14)
model_2 = Alex_net_c1()
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


#List of all the joints in the order mentioned in the dataset
joint_list = ["right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle", "right_wrist", "right_elbow", "right_shoulder", "left_shoulder", "left_elbow", "left_wrist", "neck", "head"]

#Load saved model from stage 1
model.load_state_dict(torch.load("model_1"))
pdj_thresh_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
pdj_wrist_list = []
pdj_elbow_list = []
pdj_shoulder_list = []
pdj_ankle_list = []
pdj_knee_list = []
pdj_hip_list = []
pdj_neck_list = []
pdj_head_list = []

def pcp(y, predictions):
    '''
    Computes the points per corrected parts metric for the predictions against the ground truch limb points.
    Input parameters: 
        y: Ground truth points
        predictions: Predicted Points from stage 1
    Output parameters:
        None
    '''
    predictions = predictions.reshape((predictions.shape[0],14,2))
    y = y.reshape((y.shape[0],14,2))
    upper_arm_count = 0
    lower_arm_count = 0
    upper_leg_count = 0
    lower_leg_count = 0
    
    for i in range(y.shape[0]):
        #upper arms at 0.5
        upper_arm_dist_right = np.linalg.norm(y[i, 8]-y[i, 7])
        upper_arm_dist_left = np.linalg.norm(y[i, 9]-y[i, 10])
        upper_arm_pred_right = np.linalg.norm(predictions[i, 8]-y[i,8])
        if int(upper_arm_pred_right) <= int(0.5*upper_arm_dist_right):
            upper_arm_pred_right = np.linalg.norm(predictions[i, 7]-y[i,7])
            if int(upper_arm_pred_right) <= int(0.5*upper_arm_dist_right):
                upper_arm_count +=1
        upper_arm_pred_left = np.linalg.norm(predictions[i, 9]-y[i,9])
        if int(upper_arm_pred_left) <= int(0.5*upper_arm_dist_left):
            upper_arm_pred_left = np.linalg.norm(predictions[i, 10]-y[i,10])
            if int(upper_arm_pred_left) <= int(0.5*upper_arm_dist_left):
                upper_arm_count +=1
        #Lower arms pcp at 0.5
        lower_arm_dist_right = np.linalg.norm(y[i, 7]-y[i, 6])
        lower_arm_dist_left = np.linalg.norm(y[i, 10]-y[i, 11])
        lower_arm_pred_right = np.linalg.norm(predictions[i, 7]-y[i,7])
        if int(lower_arm_pred_right) < int(0.5*lower_arm_dist_right):
            lower_arm_pred_right = np.linalg.norm(predictions[i, 6]-y[i,6])
            if int(lower_arm_pred_right) < int(0.5*lower_arm_dist_right):
                lower_arm_count +=1
        lower_arm_pred_left = np.linalg.norm(predictions[i, 10]-y[i,10])
        if int(lower_arm_pred_left) <= int(0.5*lower_arm_dist_left):
            lower_arm_pred_left = np.linalg.norm(predictions[i, 11]-y[i,11])
            if int(lower_arm_pred_left) <= int(0.5*lower_arm_dist_left):
                lower_arm_count +=1
                
        #upper legs pcp at 0.5
        upper_leg_dist_right = np.linalg.norm(y[i, 2]-y[i, 1])
        upper_leg_dist_left = np.linalg.norm(y[i, 3]-y[i, 4])
        upper_leg_pred_right = np.linalg.norm(predictions[i, 2]-y[i,2])
        if int(upper_leg_pred_right) <= int(0.5*upper_leg_dist_right):
            upper_leg_pred_right = np.linalg.norm(predictions[i, 1]-y[i,1])
            if int(upper_leg_pred_right) <= int(0.5*upper_leg_dist_right):
                upper_leg_count +=1
        upper_leg_pred_left = np.linalg.norm(predictions[i, 3]-y[i,3])
        if int(upper_leg_pred_left) <= int(0.5*upper_leg_dist_left):
            upper_leg_pred_left = np.linalg.norm(predictions[i, 4]-y[i,4])
            if int(upper_leg_pred_left) <= int(0.5*upper_leg_dist_left):
                upper_leg_count +=1
        #lower legs pcp at 0.5
        lower_leg_dist_right = np.linalg.norm(y[i, 1]-y[i, 0])
        lower_leg_dist_left = np.linalg.norm(y[i, 4]-y[i, 5])
        lower_leg_pred_right = np.linalg.norm(predictions[i, 1]-y[i,1])
        if int(lower_leg_pred_right) <= int(0.5*lower_leg_dist_right):
            lower_leg_pred_right = np.linalg.norm(predictions[i, 0]-y[i,0])
            if int(lower_leg_pred_right) <= int(0.5*lower_leg_dist_right):
                lower_leg_count +=1
        lower_leg_pred_left = np.linalg.norm(predictions[i, 4]-y[i,4])
        if int(lower_leg_pred_left) <= int(0.5*lower_leg_dist_left):
            lower_leg_pred_left = np.linalg.norm(predictions[i, 5]-y[i,5])
            if int(lower_leg_pred_left) <= int(0.5*lower_leg_dist_left):
                lower_leg_count +=1
        
    print(f'Upper arms pcp @ 0.5: {upper_arm_count/(2*predictions.shape[0])}, Lower arms pcp @ 0.5: {lower_arm_count/(2*predictions.shape[0])}, Upper legs pcp @ 0.5: {upper_leg_count/(2*predictions.shape[0])}, Lower legs pcp @ 0.5 {lower_leg_count/(2*predictions.shape[0])}')
    #print(predictions.shape, predictions)
   
def pdj(pred, Y, thresh = 0.5):
    '''
    Computes the Percentage of detected joints metric for the predictions against the ground truch limb points.
    Input parameters: 
        y: Ground truth points
        pred: Predicted Points from stage 1
        thresh: Threshold for considering whether a joint is correctly classified or not.
    Output parameters:
        None
    '''
    pdj_scores_per_part = np.zeros(8)
    for i in range(Y.shape[0]):
        dist = thresh*np.sqrt((Y[i, 6]-Y[i, 16])**2 + (Y[i, 7]-Y[i, 17])**2)
        # PDJ for ankles
        right_ankle = np.linalg.norm(np.array([pred[i,0:2]-Y[i,0:2]]))
        if right_ankle<=dist:
            pdj_scores_per_part[0]+=1/(Y.shape[0]*2)
        left_ankle = np.linalg.norm(np.array([pred[i,10:12]-Y[i,10:12]]))
        if left_ankle<=dist:
            pdj_scores_per_part[0]+=1/(Y.shape[0]*2)
        # PDJ for knees
        right_knee = np.linalg.norm(np.array([pred[i,2:4]-Y[i,2:4]]))
        if right_knee<=dist:
            pdj_scores_per_part[1]+=1/(Y.shape[0]*2)
        left_knee = np.linalg.norm(np.array([pred[i,8:10]-Y[i,8:10]]))
        if left_knee<=dist:
            pdj_scores_per_part[1]+=1/(Y.shape[0]*2)
        # PDJ for hips
        right_hip = np.linalg.norm(np.array([pred[i,4:6]-Y[i,4:6]]))
        if right_hip<=dist:
            pdj_scores_per_part[2]+=1/(Y.shape[0]*2)
        left_hip = np.linalg.norm(np.array([pred[i,6:8]-Y[i,6:8]]))
        if left_hip<=dist:
            pdj_scores_per_part[2]+=1/(Y.shape[0]*2)
        # PDJ for wrists
        right_wrist = np.linalg.norm(np.array([pred[i,12:14]-Y[i,12:14]]))
        if right_wrist<=dist:
            pdj_scores_per_part[3]+=1/(Y.shape[0]*2)
        left_wrist = np.linalg.norm(np.array([pred[i,22:24]-Y[i,22:24]]))
        if left_wrist<=dist:
            pdj_scores_per_part[3]+=1/(Y.shape[0]*2)
        # PDJ for elbows
        right_elbow = np.linalg.norm(np.array([pred[i,14:16]-Y[i,14:16]]))
        if right_elbow<=dist:
            pdj_scores_per_part[4]+=1/(Y.shape[0]*2)
        left_elbow = np.linalg.norm(np.array([pred[i,20:22]-Y[i,20:22]]))
        if left_elbow<=dist:
            pdj_scores_per_part[4]+=1/(Y.shape[0]*2)
        # PDJ for shoulders
        right_shoulder = np.linalg.norm(np.array([pred[i,16:18]-Y[i,16:18]]))
        if right_shoulder<=dist:
            pdj_scores_per_part[5]+=1/(Y.shape[0]*2)
        left_shoulder = np.linalg.norm(np.array([pred[i,18:20]-Y[i,18:20]]))
        if left_shoulder<=dist:
            pdj_scores_per_part[5]+=1/(Y.shape[0]*2)
        # PDJ for neck
        neck = np.linalg.norm(np.array([pred[i,24:26]-Y[i,24:26]]))
        if neck<=dist:
            pdj_scores_per_part[6]+=1/(Y.shape[0])
        # PDJ for ankles
        head = np.linalg.norm(np.array([pred[i,26:28]-Y[i,26:28]]))
        if head<=dist:
            pdj_scores_per_part[7]+=1/(Y.shape[0])
            
    pdj_ankle_list.append(pdj_scores_per_part[0])
    pdj_knee_list.append(pdj_scores_per_part[1])
    pdj_hip_list.append(pdj_scores_per_part[2])
    pdj_wrist_list.append(pdj_scores_per_part[3])
    pdj_elbow_list.append(pdj_scores_per_part[4])
    pdj_shoulder_list.append(pdj_scores_per_part[5])
    pdj_neck_list.append(pdj_scores_per_part[6])
    pdj_head_list.append(pdj_scores_per_part[7])

#PyTorch based custom dataset for the data
class CustomDataSet(Dataset):
    def __init__(self, data_file, data_labels):
        self.x = np.load(data_file)
        self.y = np.load(data_labels)
        a = self.x[0].shape[0]/2
        b = self.x[0].shape[1]/2
        self.y[:, 0::2] = (self.y[:, 0::2]-b)/self.x[0].shape[1]
        self.y[:, 1::2] = (self.y[:, 1::2]-a)/self.x[0].shape[0]
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        image = transform(self.x[index])
        label = torch.tensor(self.y[index])
        return image, label


test_data = CustomDataSet("./test_images.npy", "./test_labels.npy")
#Create a test data-loader using custom dataset with batch size of 96.
test_dataloader = DataLoader(test_data, batch_size=96, shuffle = False)
model = model.to(dev)
model.eval()
prediction_2 = []
#MSE Loss
loss = torch.nn.MSELoss()
x = []
y = []
#Stage 1 Test Pipeline
with torch.no_grad():
    epoch_loss = 0
    for batch_data in test_dataloader:
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
        epoch_loss += Loss.item()/(test_data.__len__()/96)
        for prediction in predictions:
            prediction_2.append(prediction.detach().cpu().numpy())
    print("Test Loss: ", epoch_loss)
x = np.array(x)
y = np.array(y)
prediction_2 = np.array(prediction_2)
prediction_2 = (prediction_2*227)+113.5
y[0::2]=(y[0::2]*227)+113.5
y[1::2]=(y[1::2]*227)+113.5

#Compute PCP
pcp(y, prediction_2)
for thresh in pdj_thresh_list:
    pdj(prediction_2, y, thresh)
    
#Plot PDJ results
plt.plot(pdj_thresh_list, pdj_ankle_list, label = "Ankles",color='red')
plt.plot(pdj_thresh_list, pdj_knee_list, label = "Knees", color='green')
plt.plot(pdj_thresh_list, pdj_hip_list, label = "Hips", color='blue')
plt.xlabel("PDJ threshold")
plt.ylabel("PDJ")
plt.title("PDJ for Legs")
plt.legend()
plt.show()

plt.plot(pdj_thresh_list, pdj_wrist_list, label = "Wrists", color='red')
plt.plot(pdj_thresh_list, pdj_elbow_list, label = "Elbows", color='green')
plt.plot(pdj_thresh_list, pdj_shoulder_list, label = "Shoulders", color='blue')
plt.xlabel("PDJ threshold")
plt.ylabel("PDJ")
plt.title("PDJ for Hands")
plt.legend()
plt.show()


#Print an example image with all the predicted points

i=600
plt.imshow(x[600,:,:,::-1])
plt.plot([prediction_2[i,0],prediction_2[i,2]], [prediction_2[i,1],prediction_2[i,3]], color="blue")
plt.plot([prediction_2[i,2],prediction_2[i,4]], [prediction_2[i,3],prediction_2[i,5]], color="blue")
plt.plot([prediction_2[i,4],prediction_2[i,6]], [prediction_2[i,5],prediction_2[i,7]], color="blue")
plt.plot([prediction_2[i,6],prediction_2[i,8]], [prediction_2[i,7],prediction_2[i,9]], color="blue")
plt.plot([prediction_2[i,8],prediction_2[i,10]], [prediction_2[i,9],prediction_2[i,11]], color="blue")
plt.plot([prediction_2[i,4],prediction_2[i,16]], [prediction_2[i,5],prediction_2[i,17]], color="blue")
plt.plot([prediction_2[i,12],prediction_2[i,14]], [prediction_2[i,13],prediction_2[i,15]], color="blue")
plt.plot([prediction_2[i,14],prediction_2[i,16]], [prediction_2[i,15],prediction_2[i,17]], color="blue")
plt.plot([prediction_2[i,6],prediction_2[i,18]], [prediction_2[i,7],prediction_2[i,19]], color="blue")
plt.plot([prediction_2[i,18],prediction_2[i,20]], [prediction_2[i,19],prediction_2[i,21]], color="blue")
plt.plot([prediction_2[i,20],prediction_2[i,22]], [prediction_2[i,21],prediction_2[i,23]], color="blue")
plt.plot([prediction_2[i,16],prediction_2[i,24]], [prediction_2[i,17],prediction_2[i,25]], color="blue")
plt.plot([prediction_2[i,18],prediction_2[i,24]], [prediction_2[i,19],prediction_2[i,25]], color="blue")
plt.plot([prediction_2[i,24],prediction_2[i,26]], [prediction_2[i,25],prediction_2[i,27]], color="blue")
plt.plot([y[i,0],y[i,2]], [y[i,1],y[i,3]], color="orange")
plt.plot([y[i,2],y[i,4]], [y[i,3],y[i,5]], color="orange")
plt.plot([y[i,4],y[i,6]], [y[i,5],y[i,7]], color="orange")
plt.plot([y[i,6],y[i,8]], [y[i,7],y[i,9]], color="orange")
plt.plot([y[i,8],y[i,10]], [y[i,9],y[i,11]], color="orange")
plt.plot([y[i,4],y[i,16]], [y[i,5],y[i,17]], color="orange")
plt.plot([y[i,12],y[i,14]], [y[i,13],y[i,15]], color="orange")
plt.plot([y[i,14],y[i,16]], [y[i,15],y[i,17]], color="orange")
plt.plot([y[i,6],y[i,18]], [y[i,7],y[i,19]], color="orange")
plt.plot([y[i,18],y[i,20]], [y[i,19],y[i,21]], color="orange")
plt.plot([y[i,20],y[i,22]], [y[i,21],y[i,23]], color="orange")
plt.plot([y[i,16],y[i,24]], [y[i,17],y[i,25]], color="orange")
plt.plot([y[i,18],y[i,24]], [y[i,19],y[i,25]], color="orange")
plt.plot([y[i,24],y[i,26]], [y[i,25],y[i,27]], color="orange")
plt.show()