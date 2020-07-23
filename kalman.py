from scipy import signal
import shutil
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
from filterpy.kalman import KalmanFilter
import helper as h
import cv2
import numpy.ma as ma
import of


def kalman(bbox, gt, stats, Q = 0):

    #define kalman filter
    nx,nz = 4,4
    Kal = KalmanFilter(dim_x = nx, dim_z = nz)
    Kal.R = np.eye(nz) * 500
    Kal.R[0,0] = stats[0]
    Kal.R[1,1] = stats[1]
    Kal.R[2,2] = stats[2]
    Kal.R[3,3] = stats[3]
    Kal.Q = np.eye(nx) * float(Q)
    Kal.P *= 0
    Kal.F = np.eye(nx)
    #Kal.F[2,4] = 1
    Kal.H = np.eye(nz,nx,0)
    init = np.zeros(nx)
    

    init[:4] = gt[0,:]
    #init[4] = 1
    
    Kal.x = init

    end = h.lastindex(gt)
    mem = ma.array(np.zeros((bbox.shape[0],nx)), mask = True)
    mem[0,:] = init
    memk = np.zeros((end,nx))
    memp = np.zeros((end,nx))

    for i in range(1, end):

        z = bbox[i, :]
        if bbox.mask[i, 0] == True:
            z = None

        Kal.predict()
        Kal.update(z)

        x = Kal.x

        mem[i,:] = x[:]
        for j in range(nx):
            memk[i,j] = Kal.K[j,j]
        for j in range(nx):
            memp[i,j] = Kal.P[j,j]

    return mem, memk,memp

def delpath():
    try:
        shutil.rmtree("./gen/kalman")
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
   
def main():

    delpath()
    h.makedirsx("./gen/kalman/")

    bbox = h.readmaarray(clippath + "/detections/clean")[:,2:6]
    gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
    end = h.lastindex(gt)

    #mu = [-16,3,12,1]  
    mu = [0,0,0,0]
    std = [17,15,18,17]

    bbox = h.createfakedet(gt,mu,std)

    #for i in range(0, bbox.shape[0], 1):
    #    bbox.mask[i,:] = True

    modelinfo = np.array([["low",1],["medium", 10],["large",100]])

    for i in range(modelinfo.shape[0]):

        mem, memk, memp = kalman(bbox,gt, std, Q = modelinfo[i, 1])
        
        ylabel = ["xmin [px]","ymin [px]","xmax [px]","ymax [px]"]
        h.timeplot3(gt,bbox, mem, modelinfo[i,0] + "-results", [[0,1920],[1080,0],[0,400],[0,2]], ylabel, ["Frame [k]"]*4, savepath + "ot" + modelinfo[i,0])
        h.plotk(memk, savepath + modelinfo[i,0])
        h.plotp(memp, savepath + modelinfo[i,0])

        memioubefore = h.iouclip(bbox,gt)
        memiouafter = h.iouclip(mem, gt)
        print(i)
        print(np.mean(memioubefore)) 
        print(np.mean(memiouafter))
        print("--------")
        print(h.rmse(gt,bbox))
        print(h.rmse(gt,mem))
        print(np.mean(memk, axis = 0))



mainvid = "Krefeld_HBF_Duisburg Ruhrort"
clipname = "straight"
clippath = "./data/" + mainvid + "/" + clipname 
savepath = "./gen/kalman/"
np.random.seed(0)
main()
