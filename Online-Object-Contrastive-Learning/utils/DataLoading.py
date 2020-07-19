#!/usr/bin/env python
# coding: utf-8

import cv2
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

class Video2Frame:
    def __init__(self, dataroot):
        self.dataroot = dataroot
    def split_video(self, video_name, seconds):
        
        vidPath = os.path.join(self.dataroot, video_name) 
        for sec in seconds:
            item_name = os.path.join(self.dataroot+"/"+"video-"+str(sec)+".mp4")
            ffmpeg_extract_subclip(vidPath, sec[0], sec[1], targetname=item_name)
            
    def read_frame(self, video_name,Frame_Fre):
        times=0
        frameFrequency=Frame_Fre
        frame_path = os.path.join(self.dataroot +"/"+ "Frame_"+video_name)
        video_path = os.path.join(self.dataroot+ "/"+video_name)
        if not os.path.exists(frame_path):
            os.mkdir(frame_path)  
        cap= cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            if times%frameFrequency == 0:
                cv2.imwrite(frame_path+"/"+"frame"+"_"+str(times)+".jpg",frame)
            times+=1
        cap.release()
        cv2.destroyAllWindows()

class Datareading(Dataset):
    def __init__(self,root):
        super(Datareading,self).__init__()
        self.root = root
    
    def data_address_reading(self):
        
        Train_images = {}
        Test_images = {}
        #folder_path = os.path.join(self.root + "/" + folder_name)
        files = os.listdir(self.root)
        for subFile in tqdm(files,ascii=True,desc="loading train&test data address"):
            if("Train" in subFile):
                file_path = open(self.root+'/'+subFile)
                iter_file = list(iter(file_path))
                for line in iter_file:
                    line = line[0:26]
                    if "\n" in line:
                        line = line.replace("\n","")
                    images_path = os.path.join(self.root +"/" + line)
                    image_items = os.listdir(images_path)
                    train_images_path = []
                    for train_image in image_items:
                        train_image_path = os.path.join(images_path + '/' + train_image)
                        train_images_path.append(train_image_path)
                    Train_images[line] = train_images_path
                    
            elif("Test" in subFile):
                file_path = open(self.root+'/'+subFile)
                iter_file = list(iter(file_path))
                for line in iter_file:
                    line = line[0:26]
                    if "\n" in line:
                        line = line.replace("\n","")
                    images_path = os.path.join(self.root + '/' + line)
                    test_image_items = os.listdir(images_path)
                    test_images_path = []
                    for test_image in test_image_items:
                        test_image_path = os.path.join(images_path + '/' + test_image)
                        test_images_path.append(test_image_path)
                    Test_images[line] = test_images_path
        return Train_images,Test_images
    
    def data_reading(self,data_address,transform = None):
        images = []
        for add in tqdm(data_address,ascii=True, desc="data is reading"):
            image = cv2.imread(add)
            image = transform(image)
            images.append(image)
        return images
            
if __name__ == "__main__":
    vid2frame = Video2Frame("./datasets")
    vid2frame.split_video("multiple-objects.mp4", seconds=[(0,5),(0,10),(0,20),(0,40),(0,80),(0,160),(160,200)])
    vid2frame.read_frame("video-(0, 5).mp4",15)
    vid2frame.read_frame("video-(0, 10).mp4",15)
    vid2frame.read_frame("video-(0, 20).mp4",15)
    vid2frame.read_frame("video-(0, 40).mp4",15)
    vid2frame.read_frame("video-(0, 80).mp4",15)
    vid2frame.read_frame("video-(0, 160).mp4",15)
    vid2frame.read_frame("video-(160, 200).mp4",15)
    
    datareading = Datareading("./datasets")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset, testset = datareading.data_address_reading()
    # for example, reading trainset["Frame_video-(0,5).mp4"]
    train_data = datareading.data_reading(trainset["Frame_video-(0,5).mp4"],transform)
    print("showing the example->>>>",train_data)
            
            
            
            