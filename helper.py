import cv2
from PIL import Image
from datetime import datetime
import numpy as np
import argparse
import sys
import os
from os import path
np.set_printoptions(threshold=sys.maxsize)

ap = argparse.ArgumentParser()
ap.add_argument("video", type=str, help = "input vid")
args = vars(ap.parse_args())

def cvi(path):
    v = cv2.VideoCapture(path)
    mem = []
    f_total = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f_total)
    for i in range(0,f_total):
        ok , frame = v.read()
        mem.append(frame)

    return np.array(mem)

def write(doc_path, data, new_line):
    
    file = open("./data/video.txt","w")
    file.write("test")
    file.write(str(number))
    file.close()

def video_array(path):
    vid = cv2.VideoCapture(path)
    fc = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    f = 0
    video = []
    
    while f<fc:
        _,frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
        f += 1
    
    return video

def array_save_frames(path):
    
    vid = extract_name(path)
    video = video_array(path)
    for i in range(len(video)):
        Image.fromarray(video[i]).save("./data/" + vid + "/data/" + vid + str(i) + ".jpg")
        #Image.fromarray(video[i]).save("./data/test/data/test.jpg")


def extract_name(path):
    vid = path.split("/")
    vid = vid[len(vid)-1]
    vid = vid.split(".")[0]
    return vid


def create_folders(video_path):
    
    vid = extract_name(video_path)
    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")
    #now = now.strftime("%Y")
    print(now)
    if not path.exists("./data/" + vid):
        os.makedirs("./data/" + vid)
        os.makedirs("./data/" + vid + "/data")
        os.makedirs("./data/" + vid + "/detbbox")
        os.makedirs("./data/" + vid + "/"+ now)
    else:
        os.makedirs("./data/" + vid + "/"+ now)
            

create_folders(args["video"])
array_save_frames(args["video"])


