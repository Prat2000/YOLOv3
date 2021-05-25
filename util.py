from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

# predict_transform transforms three feature maps of three scales into 2D tensor(intsead of 3D)
def predict_transform(prediction,inp_dim,anchors,num_classes,CUDA=True):
    batch_size=prediction.size(0)
    stride=inp_dim//prediction.size(2)
    grid_size=inp_dim//stride
    bbox_attrs=5+num_classes
    num_anchors=len(anchors)

    #view() -> changes view of a given tensor
    prediction = prediction.view(batch_size,bbox_attrs*num_anchors,grid_size*grid_size)  #dim(0) represents 3rd dim->#batches, dim(1)->#rows, dim(2)->#cols
    prediction = prediction.transpose(1,2).contiguous() #transposes rows and cols
    prediction = prediction.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs) #now each row has all anchor boxes of all cells stacked, cols->box attributes, dim(0)->#batches

    anchors=[(a[0]/stride,a[1]/stride) for a in anchors]  #reduces anchor box dim to detection map dim

    # applies sigmoid function to predictions to get bounding box coordinates wrt (0,0) as per maths
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0]) #center(x) coord
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1]) #center(y) coord
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4]) #obectiveness score

    # adding grid offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset  # grid offsets added to center coord

    # apply log space transform -> multiply the exp() to anchor attrs
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    anchors=anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # sigmoid activation to class scores
    prediction[:,:,5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))

    # resize detection map to input image size
    prediction[:,:,:4] *= stride #center coords, hgth, wdth transformed 

    return prediction  #this prediction containes final bounding box dimensions and can be drawn in the input image to generate output with bounding boxes
def write_results(prediction, confidence, num_classes, nms_conf=0.4):
        # confidence -> objectiveness score threshold
        # nms_conf -> NMS IoU threshold
        # if objectiveness score of box < confidence -> set all 85 attributes to 0
        conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2) # creates a mask with only those boxes hving obj. score>4 and unsqueezes their 2nd dimension-> here the boxes dimension
        prediction = prediction*conf_mask # only boxes with greater obj score remain, others become 0
        
        # calculating IoU-> determine top left x, top left y, bottom right x, bottom right y corners
        box_corner = prediction.new(prediction.shape)
        box_corner[:,:,0] = (prediction[:,:,0]-prediction[:,:,2]/2)
        box_corner[:,:,1] = (prediction[:,:,1]-prediction[:,:,3]/2)
        box_corner[:,:,2] = (prediction[:,:,0]+prediction[:,:,2]/2)
        box_corner[:,:,3] = (prediction[:,:,1]-prediction[:,:,3]/2)
        prediction[:,:,:4] = box_corner[:,:,:4] # replaces the original coordinates

        batch_size = prediction.size(0)
        
        write = False
        for ind in range(batch_size):
            image_pred = prediction[ind] #ind indexed image in batch -. image tensor output
            max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes],1) #find max of all probabilities -> determine class to which image belongs in dim=1
            # max_cof stores value of max conf, max_conf_score stores the index of the class
            max_conf = max_conf.float().unsqueeze(1)
            seq = (image_pred[:,:5], max_conf, max_conf_score)
            image_pred = torch.cat(seq, 1)

            non_zero_ind = torch.nonzero(image_pred[:,4]) #get rid of low objectiveness score boxes
            
            #if there are no predictions
            try:
                image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
            except:
                continue
                
            if image_pred_.shape[0]==0:
                continue
            img_classes = unique(image_pred_[:,-1]) # -1 index holds class index, gets all classes detected in image




