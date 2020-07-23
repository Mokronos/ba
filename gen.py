import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
from filterpy.kalman import KalmanFilter
import helper as h
import cv2
import numpy.ma as ma
import of
import kal

def customimagesave(clippath, framenumber, savepath):

    bbox = h.readmaarray(clippath + "/detections/raw")

    i = framenumber


    img = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i) + ".png",1)

    
    for j in range(bbox.shape[1]//6):
        if bbox.mask[i, j*6] == False:
            img = cv2.rectangle(img,(int(bbox[i,j*6+2]),int(bbox[i,j*6+3])),(int(bbox[i,j*6 + 4]),int(bbox[i,j*6 + 5])), (255,255,0),3) #yolo

    cv2.imwrite(savepath + ".png", img)

def drawsinglebbox(clippath, framenumber, savepath):

    i = framenumber

    img = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i) + ".png",1)

    bbox = cv2.selectROI(img)

    bbox = list(bbox)
    
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]

    img = cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (255,255,0),3) #yolo
    cv2.imwrite(savepath + ".png", img)


def drawgoodpoints(clippath, framenumber, savepath):

    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    i = framenumber

    img = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i) + ".png",1)

    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    points = cv2.goodFeaturesToTrack(imggray, mask = None, **feature_params)

    for i in range(points.shape[0]):

        imgpoints = cv2.circle(img, tuple(map(int,(points[i,0,:]))), 2, (255,0,255), 2)
    #img = cv2.circle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (255,255,0),3) #yolo
    cv2.imwrite(savepath + ".png", imgpoints)


#kalman filter function for testing models before implementing in main cycle
def kalman(bbox, gt, stats, std):

    #define kalman filter
    nx,nz = 4,4
    Kal = KalmanFilter(dim_x = nx, dim_z = nz)
    Kal.R = np.eye(nz) * 500
    for i in range(len(stats)):
            Kal.R[i,i] = stats[1,1,i] ** 2
    Kal.Q = np.eye(nx) * std**2
    Kal.P *= 0
    Kal.F = np.eye(nx)
    Kal.H = np.eye(nz,nx,0)

    init = np.zeros(nx)
    #calc speed for init from of

    init[:4] = gt[0,:]
    #init[4] = 1
    
    Kal.x = init

    mem = ma.array(np.zeros((bbox.shape[0],nx)), mask = True)
    mem[0,:] = init
    memk = np.zeros((bbox.shape[0]))
    memp = np.zeros((bbox.shape[0]))

    for i in range(1, bbox.shape[0]):

        #calc speed from of

        z = bbox[i, :]
        if bbox.mask[i, 0] == True:
            z = None
        Kal.predict()
        Kal.update(z)

        x = Kal.x

        mem[i,:] = x
        memk[i] = Kal.K[2,2]
        memp[i] = Kal.P[2,2]

    return mem, memk,memp



def fakekalman():
    #read coords
    gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
    statsinfo = np.load(macropath + "pre/statsdata.npy")

    gtcen = h.corcen(gt)

    fakebboxcen = h.createfakedet(gtcen,statsinfo[1,1,:])
    fakebboxcor = h.cencor(fakebboxcen)

    stdvalues = [0.01,5,25]
    for i in range(len(stdvalues)):
        #for i in range(0, bbox.shape[0], 1):
        #    bbox.mask[i,:] = True

        memcen,memk, memp = kalman(fakebboxcen,gtcen, statsinfo, stdvalues[i])
    
        memcor = h.cencor(memcen)

        memioubefore = h.iouclip(fakebboxcor,gt)
        memiouafter = h.iouclip(memcor, gt)

        print(np.mean(memioubefore)) 
        print(np.mean(memiouafter))

        ylabel = ["Mittelpunktx [px]","Mittelpunkty [px]","Breite [px]","Höhe [px]"]
        h.timeplot3(gtcen,fakebboxcen, memcen, "-results", [[0,1920],[1080,0],[0,400],[0,2]], ylabel, ["Frame [k]"]*4, savepath + clipname  + "fakedetectionsstd" + str(stdvalues[i]) )
        h.plotiou(gt, memcor, ["IoU"],  ["Frame [k]"], savepath + clipname  + "fakedetectionsstd" + str(stdvalues[i]) +  "iouot")
        h.errorplot(gt, memcor,[1,1,1,1], ylabel, ["Frame [k]"] *4, savepath + clipname  + "fakedetectionsstd" + str(stdvalues[i]) +  "error")
        h.viskal(fakebboxcor,gt,memcor,clippath, savepath + clipname + "fakedetectionsQ" + str(stdvalues[i]))
    
   
def onlydetkalman():

    textpath = "drawmaininfo1"
    #load info for clips
    mainvidpath, info = h.readtimes(textpath)
    mainvid = mainvidpath.split(".")[-2]
    mainvid = mainvid.split("/")[-1]
    datapath = "./data/"
    macropath = datapath + mainvid + "/macroanalysis/"
    savepathbase = "./gen/kalman/"

    #delete folders

    #create folders
    for i in range(np.shape(info)[0]):
        h.makedirsx(savepathbase + info[i][2])

    stdvalues = [1, 9, 81]
    statsinfo = np.load(macropath + "pre/statsdata.npy")

    memiou = []
    for i in range(len(stdvalues)):
        memiou.append([i])

    for i in range(np.shape(info)[0]):

        savepath = savepathbase + info[i][2] + "/"
        clipname = info[i][2]
        clippath = datapath + mainvid + "/" + clipname 

        #read coords
        gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
        bbox = h.readmaarray(clippath + "/detections/cleancorr")

        #define start and end where detectios appear
        start = h.firstindex(bbox)
        end = h.lastindex(bbox)

        #change gt and bbox to start at first detection 
        gt = gt[start:,:]
        bbox = bbox[start:,:]

        #transform to coordinates that are used in experiment
        gtcen = h.corcen(gt)
        bboxcen = h.corcen(bbox)


        #loop over std values to test each 
        for j in range(len(stdvalues)):

            #for i in range(0, bbox.shape[0], 1):
            #    bbox.mask[i,:] = True

            memcen,memk, memp = kalman(bboxcen,gtcen, statsinfo, stdvalues[j])
        
            memcor = h.cencor(memcen)

            memioubefore = h.iouclip(bbox,gt)
            memiouafter = h.iouclip(memcor, gt)

            print(np.mean(memioubefore)) 
            print(np.mean(memiouafter))

            memiou[j].extend(h.iouclip(memcor, gt))

            ylabel = ["Mittelpunktx [px]","Mittelpunkty [px]","Breite [px]","Höhe [px]"]
            h.timeplot3(gtcen,bboxcen, memcen, "-results", [[0,1920],[1080,0],[0,400],[0,2]], ylabel, ["Frame [k]"]*4, savepath + clipname  + "fakedetectionsstd" + str(stdvalues[j]) )
            h.plotiou(gt, memcor, ["IoU"],  ["Frame [k]"], savepath + clipname  + "fakedetectionsstd" + str(stdvalues[j]) +  "iouot")
            h.errorplot(gt, memcor,[1,1,1,1], ylabel, ["Frame [k]"] *4, savepath + clipname  + "fakedetectionsstd" + str(stdvalues[j]) +  "error")
            h.viskal(bbox,gt,memcor,clippath, savepath + clipname + "bbox" + str(stdvalues[j]), start = start)

    for j in range(len(stdvalues)):

        np.savetxt(savepathbase + "std" + str(stdvalues[j]) + "iou.txt", np.array(memiou[j][1:]),fmt = "%1.2f",header ="iou of all clips(compared at places where yolo originally detected a bbox)")
        np.savetxt(savepathbase + "std" + str(stdvalues[j]) + "iouavg.txt", [np.array(memiou[j][1:]).mean(),np.array(memiou[j][1:]).std()],fmt = "%1.2f",header ="iou of all clips(compared at places where yolo originally detected a bbox)")

#file to quickly try stuff or create plots needed fast
#define clip to run stuff over
mainvid = "Krefeld_HBF_Duisburg Ruhrort"
datapath = "./data/"
macropath = datapath + mainvid + "/macroanalysis/"
clipname = "rightcurve"
clippath = "./data/" + mainvid + "/" + clipname 
savepath = "./gen/kalman/"

framenumber = 68
#customimagesave(clippath, framenumber, "./gen/" + clipname + str(framenumber) + "yolodet")
framenumber2 = 45
#drawsinglebbox(clippath, framenumber2, "./gen/" + clipname + str(framenumber2) + "signside2")
framenumber3 = 60
#drawgoodpoints(clippath,framenumber3,"./gen/" + clipname + str(framenumber3) + "goodpointstotrack")

#x = np.array([1,2,3,4,5,4])
#y = np.array([1,1,1,1,1,1])
#print(h.calcrmse(x,y))
#gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
#print(h.calcprocessstd(h.corcen(gt)))

#onlydetkalman()

