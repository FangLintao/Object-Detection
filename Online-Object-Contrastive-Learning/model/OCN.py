#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import torch.functional as f
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet50
import numpy as np
import torch

class OCN(nn.Module):
    def __init__(self, ocn_feature = 16, faster_feature=91, freeze_FasterRCNN = False,criterion = 0.5):
        super(OCN,self).__init__()
        self.freeze_FasterRcnn = freeze_FasterRCNN
        self.criterion = criterion
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        if not self.freeze_FasterRcnn:
            for param in model.parameters():
                param.requires_grad = False

        resnet = resnet50(pretrained=True)
        
        self.faster_model = nn.Sequential(model)
        self.ResNet50 = nn.Sequential(*list(resnet.children())[0:-2])
        
        self.layer1 = self.layer()
        self.layer2 = self.layer()
        self.layer3 = self.layer()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.faster_fc = nn.Linear(in_features=2048, out_features=faster_feature)
        self.ocn_fc = nn.Linear(in_features=2048, out_features=ocn_feature)
    
    def forward(self,data):
        self.faster_model.eval()                
        output = self.faster_model([data])
        
        boxes = output[0]["boxes"]
        labels = output[0]["labels"]
        scores = output[0]["scores"]
 
        Objects,boxes,labels = self.roi_pooling(data,boxes,labels,scores)
        ocn_features = []
        ce_features = []
        for item in Objects:
            item = item.unsqueeze(0)
            output = self.ResNet50(item)
            res = output
            output = self.layer1(output)
            output = res + output

            res = output
            output = self.layer2(output)
            output = res + output

            res = output
            output = self.layer3(output)
            output = res + output

            output = self.avg_pool(output)
            
            faster_output = torch.flatten(output,1)
            faster_output = self.faster_fc(faster_output)
            ce_features.append(faster_output)
            
            ocn_output = torch.flatten(output,1)
            ocn_output = self.ocn_fc(ocn_output)
            ocn_features.append(ocn_output)
            
        ce_features = torch.stack(ce_features).squeeze()
        ocn_features = torch.stack(ocn_features).squeeze()
        return ce_features, ocn_features,boxes,labels
    
    def roi_pooling(self,Image,boxes,labels,scores): 
        output = []
        bounding_box = []
        Labels = []
        for idx in range(boxes.shape[0]):
            if scores[idx] >= self.criterion:
                bounding_box.append(boxes[idx])
                Labels.append(labels[idx])
                x1, y1, x2, y2 = int(np.floor(boxes[idx][0].item())), int(np.floor(boxes[idx][1].item())), int(np.ceil(boxes[idx][2].item())), int(np.ceil(boxes[idx][3].item()))
                img = Image[:, y1:y2, x1:x2]
                output.append(img)
        return output,bounding_box, Labels
            
    def layer(self):
        return nn.Sequential(nn.Conv2d( in_channels=2048, out_channels=2048, kernel_size=3 , stride=1, padding=1 ),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d( in_channels=2048, out_channels=2048, kernel_size=3 , stride=1, padding=1 ),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d( in_channels=2048, out_channels=2048, kernel_size=3 , stride=1, padding=1 ),
                        nn.BatchNorm2d(2048),
                        nn.ReLU()
                        )