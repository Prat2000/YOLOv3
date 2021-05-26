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

def unique(tensor):  #used to select one true detection per class present in any image instead of many
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

#calculate IoU for box in Ist arg with each of the boxes in 2nd arg
def bbox_iou(box1, box2):
    # coords of 2 boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #coords of intersection rect
    inter_rect_x1 = torch.max(b1_x1, b2_x1) #top_left corner
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2) #bottom_right corner
    inter_rect_y2 = torch.min(b1_y2, b2_y2)   

    #intersection area
    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1, min=0)*torch.clamp(inter_rect_y2-inter_rect_y1+1,min=0)
    #union area
    b1_area = (b1_x2-b1_x1+1)*(b1_y2-b1_y1+1)
    b2_area = (b2_x2-b2_x1+1)*(b2_y2-b2_y1+1)
    
    iou=inter_area/(b1_area+b2_area-inter_area)

    return iou

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
            
            #instead 80 class scores add only max score and index of it
            max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes],1) #find max of all probabilities -> determine class to which image belongs in dim=1
            # max_cof stores value of max conf, max_conf_score stores the index of the class
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
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

            for cls in img_classes: # cls represents a class amongst all classes detected
                # NMS used

                #get detection with one particular class denoted by cls
                cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1) #creates mask for the class
                class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze() #only the nonzero rows remain -> squeezed
                image_pred_class = image_pred_[class_mask_ind].view(-1,7) #flattens the tensor

                # sort -> max objectiveness score object on top
                conf_sort_index = torch.sort(image_pred_class[:,4],descending=True)[1]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.size(0) # no. of detections

                for i in range(idx):
                    #get IoU for all boxes
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:]) #bbox_iou -> return iou of 1st box with each box in 2nd arg
                    except ValueError: #if no more boxes are left in 2nd arg
                        break
                    except IndexError: #if no boxes are left -> both errors imply the checking is done and we can stop NMS
                        break

                    # zero all detections with IoU > thershold -> too similar match
                    iou_mask = (ious< nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask

                    #remove nonzero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

                batch_ind = image_pred_class.new(image_pred_class.size(0),1).fill_(ind)
                seq = batch_ind, image_pred_class

                if not write:
                    output = torch.cat(seq,1)
                    write =True
                else:
                    out = torch.cat(seq,1)
                    output = torch.cat((output,out))  #output tensor contains all bounding boxes that predict all different classes

        try:
            return output  # if there is any detection then output tensor has been initialized
        except:
            return 0  # means no detection in the image -> so no result

def load_classes(namesfile): # returns a dictionary with class index and corresponding names
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def letterbox_image(img, inp_dim): # this function resizes the images and pads left out areas with color
    # resize image with unchanged aspect ratio using padding
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    # prepares image as per pytorch requirement
    # convert BGR to RGB, brings channel dimension to first dimension
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
    






