from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EmptyLayer(nn.Module):  # for defining route layer
    def __init__(self):
        super(EmptyLayer,self).__init__()

class DetectionLayer(nn.Module):  # for defining YOLO detection layer
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors=anchors

class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.blocks=parse_cfg(cfgfile)
        self.net_info, self.module_list=create_modules(self.blocks)
    # we define the forward function to get output
    def forward(self,x,CUDA):  # x -> input, CUDA -> true if GPU avl
        modules=self.blocks[1:]
        outputs={} #output of some layers are requires by route layer so we store them
        write=0
        for i, module in enumerate(modules):
            module_type=(module["type"])
            if module_type=="convolutional" or module_type=="upsample":
                x=self.module_list[i](x)  #that particular module in nn acts on the input
            elif module_type=="route":
                layers=module["layers"]
                layers=[int(a) for a in layers]

                if(layers[0]>0):
                    layers[0]=layers[0]-i
                if len(layers)==1:
                    x=outputs[i+(layers[0])]
                else:
                    if (layers[1])>0:
                        layers[1]=layers[1]-i
                    
                    map1=outputs[i+layers[0]]
                    map2=outputs[i+layers[1]]

                    x=torch.cat((map1,map2),1)
                
            elif module_type=="shortcut":
                from_=int(module["from"])
                x=outputs[i-1]+outputs[i+from_]                



# cfg file contains values of all parameters used in original paper of YOLOv3
# we use these parameters directly using the cfg file(configuration file)

def parse_cfg(cfgfile):  #to parse the cfg file
    file = open(cfgfile, 'r')
    lines = file.read().split('\n') #split into lines
    lines = [x for x in lines if len(x)>0] #remove empty lines
    lines = [x for x in lines if x[0]!='#'] #remove comments
    lines = [x.rstrip().lstrip() for x in lines] #remove whitespace
 
 # we have extracted lines from the cfg file
 # now we create blocks for each type of layer
    
    block = {}
    blocks = []
    for line in lines:
        if line[0]=="[":
            if len(block)!=0:
                blocks.append(block)
                block={}
            block["type"] = line[1:-1].rstrip() 
        else:
            key,value = line.split("=")
            block[key.rstrip()]=value.lstrip()
    blocks.append(block)
    return blocks   

def create_modules(blocks): # creates modules obtained from parse_cfg
    net_info = blocks[0] #zeroeth blcok -> net; containes overall parameters
    module_list = nn.ModuleList() #used to store list of modules similar to nn.sequential
    prev_filters=3  # stores how many filters were there in prev layer -> necssary in convolutional layer
    output_filters=[] #route layer concatenates diff convolutional layers so we store all the filters as they come and not only the prev one

# we have made initial arrangements
# now we create Pytorch modules for each layer  
    
    for index,x in enumerate(blocks[1:]):  #except the net block -> it's not a layer
        
        module= nn.Sequential()

        # convolutional layer parameters        
        if(x["type"]=="convolutional"):
            activation=x["activation"]
            try:
                batchnormalize=int(x["batchnormalize"])
                bias=False
            except:
                batchnormalize=0
                bias=True
            filters=int(x["filters"])
            padding=int(x["pad"])
            kernel_size=int(x["size"])
            stride=int(x["stride"])

            if padding:
                pad=(kernel_size-1)//2
            else:
                pad=0

            # now add convolutional layer Pytorch module

            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(index),conv)
            
            # add batchnormalize layer

            if batchnormalize:
                bn = nn.BtachNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)
            
            # now add activation layer

            if activation=="leaky":
                activn=nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky_{0}".format(index),activn)

        # upsampling layer parameters
        elif(x["type"]=="upsample"):
            stride=int(x["stride"])
            upsample=nn.Upsample(scale_factor=2,mode="bilinear") #scale factor =2 has been used in the original factor
            module.add_module("upsample_{0}".format(index),upsample)

        # route layer -> this layer concatenates two previous layers
        elif(x["type"]=="route"):
            x["layers"]=x["layers"].split(',')

            start=int(x["layers"][0])
            try:
                end=int(x["layers"][1]) #fails if layer has 1 parameter
            except:
                end=0
            
            if start>0:
                start=start-index # that many layers up
            if end>0:
                end=end-index
            
            route=EmptyLayer()
            module.add_module("route_{0}".format(index),route)

            if end<0:
                filters=output_filters[index+start] + output_filters[index+end]  #start(end) + index gives which layer to begin(end) with
            else:
                filters=output_filters[index+start]
            
        # add shortcut/ skip connection 
        elif x["type"]=="shortcut":
            shortcut=EmptyLayer()
            module.add_module("shortcut_{}".format(index),shortcut)

        # YOLO detection layer
        elif x["type"]=="yolo":
            mask=x["mask"].split(",")
            mask=[int(x) for x in mask]

            anchors=x["anchors"].split(",")
            anchors=[int(x) for x in anchors] # makes a list of all values
            anchors=[(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)] # makes pairs as given in cfg file
            anchors=[anchors[i] for i in mask]  # there are 9 pairs of sizes -> 3 scales and 3 anchor boxes per grid scale-> chosen randomly in original paper

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index),detection)
        
        module_list.append(module) # stores the entire module
        prev_filters=filters # final filter size
        output_filters.append(filters)
    
    return (net_info, module_list)

# blocks=parse_cfg("cfg/yolov3.cfg")
# print(create_modules(blocks))