import cv2
from PIL import Image
from datetime import datetime
import numpy as np
import argparse
import sys
import os
from os import path
np.set_printoptions(threshold=sys.maxsize)

#ap = argparse.ArgumentParser()
#ap.add_argument("--video", type=str, help = "input vid")
#args = vars(ap.parse_args())

def cvi(vpath):
    v = cv2.VideoCapture(vpath)
    mem = []
    f_total = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f_total)
    for i in range(0,f_total):
        ok , frame = v.read()
        mem.append(frame)

    return np.array(mem)

def video_array(vpath):
    vid = cv2.VideoCapture(vpath)
    fc = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    f = 0
    video = []
    
    while f<fc:
        _,frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
        f += 1
    
    return video

# bug: first frame gets saved twice + multiple frames more often. issue with framerate, no bug.
# need to cut videos same framerate as original. (test.mp4 -- 25fps)
def save_frames(vpath):
    
    vid = extract_name(vpath)
    video = video_array(vpath)
    for i in range(len(video)):
        Image.fromarray(video[i]).save("./data/" + vid + "/data/" + vid + str(i) + ".jpg")
        #Image.fromarray(video[i]).save("./data/test/data/test.jpg")


def extract_name(vpath):
    vid = vpath.split("/")
    vid = vid[len(vid)-1]
    vid = vid.split(".")[0]
    return vid


def create_folders(vpath):
    
    vid = extract_name(vpath)
    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    #now = now.strftime("%Y")
    if not path.exists("./data/" + vid):
        os.makedirs("./data/" + vid)
        os.makedirs("./data/" + vid + "/data")
        os.makedirs("./data/" + vid + "/detbbox")
        os.makedirs("./data/" + vid + "/groundtruth")
        os.makedirs("./data/" + vid + "/"+ now)
    else:
        os.makedirs("./data/" + vid + "/"+ now)
    return now

def video_path(video_name):
    return ("./data/" + str(video_name))

def create_txt(text, path, name):
    with open(path + "/" + name + ".txt", "a") as file:
        file.write(text + "\n")

#ugly
def arr_str(arr):
    text = ""
    if len(arr.shape)==1:
        for i in range(len(arr)):
            text = text + str(arr[i]) + " "
    else:
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                text = text + str(arr[i][j]) + " "
    return text


def frame_path(vpath, frame_number):
    return video_path(extract_name(vpath)) + "/data/" + extract_name(vpath) + str(frame_number) + ".jpg"

def frame_array(vpath, frame_number):
    return cv2.imread(frame_path(vpath,frame_number))

def get_total_frames(vpath):
    return int(cv2.VideoCapture(vpath).get(cv2.CAP_PROP_FRAME_COUNT))


# make single methods for yolo detection and for slef labeling(ground truth), otherwise folder seletion isdifficult

def self_label(vpath, cla):
    savepath = video_path(extract_name(vpath))
    print("test")
    for i in range(get_total_frames(vpath)):
        print("test1")
        frame = frame_array(savepath, i)
        print(np.array(frame).shape)
        bbox = cv2.selectROI(frame)
        bbox = np.array(bbox)
        bbox = np.insert(bbox,0,1)
        bbox = np.insert(bbox,0,cla)
        bbox = np.insert(bbox,0,i)
        create_txt((arr_str(bbox)),savepath + "/groundtruth","label")
        print(bbox)
        


#selectroi returns [xmin, ymin, xmax, ymax] from top left corner
#detmethod returns [label, confidence, x_min, y_min, x_max, y_max]
def det_bbox(det_method, vpath):
    
    #for i in range(get_total_frames(vpath)):
    #    frame = frame_array(vpath, i)
    #    #det = det_method(frame)
    #    det = [i,i,4,1,2,3]
    #    det = det.flatten()
    #    det = np.array(det)
    #    #create_txt(arr_str(det), vpath, det_method.__name__)
    #    create_txt(arr_str(det), vpath, "geg")
    for i in range(11):
        #frame = frame_array(vpath, i)
        #det = det_method(frame)
        det = [i,i,4,1,2,3]
        det = np.array(det)
        det = det.flatten()
        #create_txt(arr_str(det), vpath, det_method.__name__)
        create_txt(arr_str(det), video_path(extract_name(vpath)) + "/detbbox", "geg")


# def det_yolo(vpath): maybe in here??



    
    

if __name__ == "__main__":
    
    bbox = [224,311,4,21]
    bbox1 = np.array([1,2,3,4])
    #det_bbox(bbox,"./data/video")
    print(get_total_frames("./tdata/test.mp4"))

    #with open("text.txt", "a") as file:
    #    file.write(arr_str(bbox) + "\n")


    self_label("./tdata/test_Trim_Trim.mp4", 0)
    #create_folders("./tdata/test_Trim_Trim.mp4")
    #save_frames("./tdata/test_Trim_Trim.mp4")


