import cv2
import scipy.stats as stats
import math
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from os import path
np.set_printoptions(threshold=sys.maxsize, suppress=True)
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
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
        f += 1
    
    return video

# bug: first frame gets saved twice + multiple frames more often. issue with framerate, no bug.
# need to cut videos same framerate as original. (test.mp4 -- 25fps)
def save_frames(vpath):
    
    vid = extract_name(vpath)
    video = video_array(vpath)
    for i in range(len(video)):
#        Image.fromarray(video[i]).save("./data/" + vid + "/data/" + vid + str(i) + ".jpg")
        cv2.imwrite("./data/" + vid + "/data/" + vid + str(i) + ".png", video[i])
        #Image.fromarray(video[i]).save("./data/test/data/test.jpg")


def extract_name(vpath):
    if "/" in vpath:
        vid = vpath.split("/")
    elif "\\" in vpath:
        vid = vpath.split("\\")

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
        os.makedirs("./data/" + vid + "/tracker_prediction")
        os.makedirs("./data/" + vid + "/comparison_error")
        os.makedirs("./data/" + vid + "/tracker_prediction/" + now)
    else:
        os.makedirs("./data/" + vid + "/tracker_prediction/" + now)
    return now

def video_path(video_name):
    return ("./data/" + str(video_name))

def create_txt(text, path, name):
    with open(path + "/" + name + ".txt", "a") as file:
        file.write(text + "\n")

#ugly
def arr_str(arr):
    text = ""
    arr = np.array(arr)
    if len(arr.shape)==1:
        for i in range(len(arr)):
            text = text + str(arr[i]) + " "
    else:
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                text = text + str(arr[i][j]) + " "
    return text


def frame_path(vpath, frame_number):
    return video_path(extract_name(vpath)) + "/data/" + extract_name(vpath) + str(frame_number) + ".png"

def frame_array(vpath, frame_number):
    return cv2.imread(frame_path(vpath,frame_number))

def get_total_frames(vpath):
    return int(cv2.VideoCapture(vpath).get(cv2.CAP_PROP_FRAME_COUNT))


# make single methods for yolo detection and for slef labeling(ground truth), otherwise folder seletion isdifficult

def self_label(vpath, clas):
    savepath = video_path(extract_name(vpath))
    old_bbox=[0,0,0,0]
    
    for i in range(get_total_frames(vpath)):
        cla = [clas]
        conf = [1]
        frame = frame_array(savepath, i)
        print(old_bbox)
        frame = cv2.rectangle(frame, (old_bbox[0],old_bbox[1]), (old_bbox[2],old_bbox[3]), (0,255,0), 1)
        frame = cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)
        bbox = cv2.selectROI(frame)
        bbox = list(bbox)
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        old_bbox = bbox
        bbox = [bbox]
        if bbox == [[0,0,0,0]]:
            bbox = [[]]
            cla = []
            conf = []
        print(bbox)
        print(cla)
        print(conf)
        write_bbox(cla, conf, bbox, i, savepath + "/groundtruth")
        


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

def write_bbox(cla, conf, bbox, frame_nbr, txtpath):
    text = "{} {}".format(frame_nbr, len(cla))
    conf_round = []
    for i in range(len(cla)):
        conf_round.append(round(conf[i],4))
        
        text = text + " {} ".format(cla[i]) +str(conf_round[i])+" {} {} {} {}".format(int(round(bbox[i][0])),int(round(bbox[i][1])),int(round(bbox[i][2])),int(round(bbox[i][3])))
    create_txt(text, txtpath, "bbox") 


def read_bbox(txt_path):
    
    f = open(txt_path)
    f = f.read()
    f = f.splitlines()
    max_cla = []
    for i in range(len(f)):
        max_cla.append(f[i].split()[1])
    cla = np.zeros((len(f),int(max(max_cla))+1))
    conf = np.zeros((len(f),int(max(max_cla))+1))
    bbox = np.zeros((len(f),int(max(max_cla))+1,4))
    for i in range(len(f)):
        g = f[i].split()
        for j in range(len(g)):
            g[j] = float(g[j])
        cla[i][0] = g[1]
        conf[i][0] = g[1]
        for j in range(int(g[1])):
            cla[i][j+1] = g[2 + 6*j]
            conf[i][j+1] = g[3 + 6*j]
            for k in range(4):
                bbox[i][0][k] = g[1]
                bbox[i][j+1][k] = g[4 + (j * 6) + k]
    
#   print(cla)
#   print(conf)
#   print(bbox)
    return cla, conf, bbox

def trans_err_50(data):
    x = 5
    y = 10
    f = np.zeros(2*y)
    t = 0
    for i in range(-y,y-1):
        for j in range(len(data)):
            if i*x<=data[j]<(i+1)*x:
                f[t] += 1
        t += 1

    #x_axis = range(-y,y)
    #plt.plot(x_axis, f, "ro")
    #plt.bar(x_axis, f)
    #plt.show()

    return f


#works only for 1 class
def compare_bbox(vpath):
    vn = extract_name(vpath)

    cla1, conf1, bbox1 = read_bbox("./data/" + str(vn) + "/groundtruth/bbox.txt")
    cla2, conf2, bbox2 = read_bbox("./data/" + str(vn) + "/detbbox/bbox.txt")

    err_bbox_xmin = []
    err_bbox_ymin = []
    err_bbox_xmax = []
    err_bbox_ymax = []
    err_conf = []
    for i in range(cla1.shape[0]):
        if cla1[i][0] != 0 and cla2[i][0] != 0:
            err_bbox_xmin.append(bbox1[i][1][0] - bbox2[i][1][0])
            err_bbox_ymin.append(bbox1[i][1][1] - bbox2[i][1][1])
            err_bbox_xmax.append(bbox1[i][1][2] - bbox2[i][1][2])
            err_bbox_ymax.append(bbox1[i][1][3] - bbox2[i][1][3])
            err_conf.append(conf1[i][1] - conf2[i][1])
            #print(i)
    if not os.path.exists(video_path(vn) + "/comparison_error/xmin.txt"):

        create_txt(arr_str(err_bbox_xmin), video_path(vn) + "/comparison_error", "xmin")
        create_txt(arr_str(err_bbox_ymin), video_path(vn) + "/comparison_error", "ymin")
        create_txt(arr_str(err_bbox_xmax), video_path(vn) + "/comparison_error", "xmax")
        create_txt(arr_str(err_bbox_ymax), video_path(vn) + "/comparison_error", "ymax")
        create_txt(arr_str(err_conf),video_path(vn) + "/comparison_error", "conf")
     
    #print(err_bbox_xmin)
    #print(err_bbox_ymin)
    #print(err_bbox_xmax)
    #print(err_bbox_ymax)

    plot_hist(err_bbox_xmin,err_bbox_ymin,err_bbox_xmax,err_bbox_ymax,err_conf)


def mu_sigma(data):
    mu = np.mean(data)
    sigma = np.var(data)
    return mu, sigma**0.5

def gauss_f(x,mu,sigma):
    y = (1/(sigma*(2*math.pi)**0.5)) * np.exp((-1/2)*((x-mu)/sigma)**2)
    return y
 

def plot_hist(err_bbox_xmin,err_bbox_ymin,err_bbox_xmax,err_bbox_ymax,err_conf):

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=3.0)
    total = len(err_bbox_xmin)
    print(total)

    x = np.arange(-60,60,0.01)
    mu1,sigma1 = mu_sigma(err_bbox_xmin)
    mu2,sigma2 = mu_sigma(err_bbox_ymin)
    mu3,sigma3 = mu_sigma(err_bbox_xmax)
    mu4,sigma4 = mu_sigma(err_bbox_ymax)
    
    axs[0,0].hist(err_bbox_xmin,bins = 20, density = True, range = (-60,60), histtype = "barstacked" )
    axs[0,0].plot(x, gauss_f(x,mu1,sigma1),"r")
    axs[0,0].set_xticks(np.arange(-60,61,10))
    #axs[0,0].set_yticks(np.arange(0,1,0.1))
    #axs[0,0].set_yticklabels(np.round(np.arange(0,21,5)/total,2))
    axs[0,0].set_title("x_min")
    axs[0,0].set(xlabel="Fehler in Pixel", ylabel="normierte Fehleranzahl")

    axs[0,1].hist(err_bbox_ymin,bins = 20, density = True, range = (-60,60), histtype = "barstacked" )
    axs[0,1].plot(x, gauss_f(x,mu2,sigma2),"r")
    axs[0,1].set_xticks(np.arange(-60,61,10))
    #axs[0,1].set_yticks(np.arange(0,21,5))
    #axs[0,1].set_yticklabels(np.round(np.arange(0,21,5)/total,2))
    axs[0,1].set_title("y_min")
    axs[0,1].set(xlabel="Fehler in Pixel", ylabel="normierte Fehleranzahl")

    axs[1,0].hist(err_bbox_xmax,bins = 20, density = True, range = (-60,60), histtype = "barstacked" )
    axs[1,0].plot(x, gauss_f(x,mu3,sigma3),"r")
    axs[1,0].set_xticks(np.arange(-60,61,10))
    #axs[1,0].set_yticks(np.arange(0,21,5))
    #axs[1,0].set_yticklabels(np.round(np.arange(0,21,5)/total,2))
    axs[1,0].set_title("x_max")
    axs[1,0].set(xlabel="Fehler in Pixel", ylabel="normierte Fehleranzahl")

    axs[1,1].hist(err_bbox_ymax,bins = 20, density = True, range = (-60,60), histtype = "barstacked" )
    axs[1,1].plot(x, gauss_f(x,mu4,sigma4),"r")
    axs[1,1].set_xticks(np.arange(-60,61,10))
    #axs[1,1].set_yticks(np.arange(0,21,5))
    #axs[1,1].set_yticklabels(np.round(np.arange(0,21,5)/total,2))
    axs[1,1].set_title("y_max")
    axs[1,1].set(xlabel="Fehler in Pixel", ylabel="normierte Fehleranzahl")
    
    fig2 = plt.figure()
    plt.hist(err_conf,bins = 30, range = (0,0.6), histtype = "barstacked" )
    plt.xticks(np.arange(0,0.6,0.1))
    plt.yticks(np.arange(0,71,10),np.round(np.arange(0,71,10)/total,2))
    plt.title("Konfidenzenverteilung")
    plt.xlabel("Konfidenzfehler")
    plt.ylabel("normierte Fehleranzahl")
    #plt.text(-100, 80 , "total datapoints: " + str(len(err_bbox_xmin)), fontsize=12)
    print(stats.norm.fit(err_bbox_xmin))
    print(mu1)
    print(sigma1)
    #fig3 = plt.figure()
    #plt.plot(x,gauss_f(x,mu1,sigma1))
    plt.show()


def read_arr(txt_path):
    
    f = open(txt_path, "r")
    f = f.read()
    f = f.split(" ")
    f = f[0:len(f)-1]
    f = list(map(float, f))
    return f

def plot_all():
    l = os.listdir("./data")
    #l.remove("stats")
    xmin = np.array([])
    ymin = np.array([])
    xmax = np.array([])
    ymax = np.array([])
    conf = np.array([])
    for i in range(len(l)):
        
        xmin = np.append(xmin, read_arr("./data/" + l[i] + "/comparison_error/xmin.txt"))
        ymin = np.append(ymin, read_arr("./data/" + l[i] + "/comparison_error/ymin.txt"))
        xmax = np.append(xmax, read_arr("./data/" + l[i] + "/comparison_error/xmax.txt"))
        ymax = np.append(ymax, read_arr("./data/" + l[i] + "/comparison_error/ymax.txt"))
        conf = np.append(conf, read_arr("./data/" + l[i] + "/comparison_error/conf.txt"))
        
    #print(xmin)
    
    plot_hist(xmin, ymin, xmax, ymax,conf)
    #x.append(read_arr(""))

    

# def det_yolo(vpath): maybe in here??
    
def plot_over_time(vpath):
    
    vn = extract_name(vpath)
    cla1, conf1, bbox1 = read_bbox("./data/" + str(vn) + "/groundtruth/bbox.txt")
    cla2, conf2, bbox2 = read_bbox("./data/" + str(vn) + "/detbbox/bbox.txt")

    fmax = cla1.shape[0]
    ex = 0
    start = 0
    err_bbox = np.zeros((fmax,5))
     
    print(err_bbox.shape)
    for i in range(fmax):
        if cla1[i,0] != 0 and cla2[i,0] != 0:
            err_bbox[i,0:4] = bbox1[i,1,:] -bbox2[i,1,:]
            if ex == 0:
               start = i
            ex = 1
        if cla1[i,0] != 0 and cla2[i,0] != 0:
            end = i
        if cla1[i,0] == 0 or cla2[i,0] == 0:
            err_bbox[i,0:4] = None
            bbox2[i,1,:] = bbox1[i,1,:]

    for i in range(fmax):
        if cla1[i,0] != 0 and cla2[i,0] != 0:
            break
    
    tick_y_max = np.amax(err_bbox)
    print(err_bbox)
    tick_y_min = np.amin(err_bbox)
    print(tick_y_max)
    x = np.arange(0,end-start,1)
    x2 = np.arange(0,fmax)



    #plt.plot(x, bbox1[start:end,1,0]-bbox2[start:end,1,0])
    #plt.title("links")
    #plt.figure()
    #plt.plot(x, bbox1[start:end,1,1]-bbox2[start:end,1,1])
    #plt.title("oben")
    #plt.figure()
    #plt.plot(x, bbox1[start:end,1,2]-bbox2[start:end,1,2])
    #plt.title("rechts")
    #plt.figure()
    #plt.plot(x, bbox1[start:end,1,3]-bbox2[start:end,1,3])
    #plt.title("unten")

    #plt.show()


    #plt.figure()
    err_bbox = err_bbox.astype(np.double)
    err_mask = np.isfinite(err_bbox)
    plt.plot(x2[err_mask[:,0]],err_bbox[err_mask[:,0],0], marker= "o" )
    plt.yticks(np.arange(tick_y_min,tick_y_max,5))
    plt.title("links")
    plt.figure()
    plt.plot(x2[err_mask[:,1]],err_bbox[err_mask[:,1],1], marker= "o" )
    plt.yticks(np.arange(tick_y_min,tick_y_max,5))
    plt.title("oben")
    plt.figure()
    plt.plot(x2[err_mask[:,2]],err_bbox[err_mask[:,2],2], marker= "o" )
    plt.yticks(np.arange(tick_y_min,tick_y_max,5))
    plt.title("rechts")
    plt.figure()
    plt.plot(x2[err_mask[:,3]],err_bbox[err_mask[:,3],3], marker= "o" )
    plt.yticks(np.arange(tick_y_min,tick_y_max,5))
    plt.title("unten")
    plt.show()
    

if __name__ == "__main__":
    
    bbox = [224,311,4,21]
    bbox1 = np.array([1,2,3,4])
    #det_bbox(bbox,"./data/video")
    print(get_total_frames("./tdata/test.mp4"))

    #with open("text.txt", "a") as file:
    #    file.write(arr_str(bbox) + "\n")
    #compare_bbox(r"C:\Users\Sebastian\Videos\1_58.mp4")
    #print(read_arr("./data/bahn_1s/comparison_error/xmin.txt"))
    plot_over_time("./clips/1_58.mp4")
    #self_label(r"C:\Users\Sebastian\Videos\5_34.mp4", 0)
    #plot_all()
    #create_folders("./tdata/test_Trim_Trim.mp4")
    #save_frames("./tdata/test_Trim_Trim.mp4")


