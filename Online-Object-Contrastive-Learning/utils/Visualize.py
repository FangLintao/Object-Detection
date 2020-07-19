#!/usr/bin/env python
# coding: utf-8

import cv2
import torch


class Visualize:
    def __init__(self):
        pass
    def OCN_visualize(self,image, boxes):
        if image.shape[0] == 3:
            image = torch.transpose(image,0,1)
            image = torch.transpose(image,1,2)
        image = image.numpy()
        for bb in boxes:
            x1 = int(torch.floor(bb[0]).item())
            y1 = int(torch.floor(bb[1]).item())
            x2 = int(torch.floor(bb[2]).item())
            y2 = int(torch.floor(bb[3]).item())
            cv2.rectangle(image, (x1,y1),(x2,y2), (255,255,0),2)
        cv2.imshow('Image',image)
        cv2.waitKey(0)