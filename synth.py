import cv2
import matplotlib.pyplot as plt
import os
import numpy.ma as ma
import numpy as np
import helper as h

#read images from folder --> create video --> save in right folder
#path to synthetic data (images)
path = "./sourcedata/synthetic/Acceleration_Normal/Normal/"
videosavepath = "./sourcedata/sourcevideos/"
genpath = "./gen/"
def createvid(path, videosavepath):

    #extract vidname
    vidname = path.split("/")[3]
    #extract imagename
    samplelist = os.listdir(path)
    imagelist = []
    for i in range(len(samplelist)):
        if samplelist[i].split(".")[-1] == "jpg":
            imagelist.append(samplelist[i])

    #create videowriter
    video = cv2.VideoWriter(videosavepath + vidname + ".avi",cv2.VideoWriter_fourcc(*"XVID"),30,(1920,1080))

    for i in range(len(imagelist)):
        frame = cv2.imread(path + imagelist[i])
        video.write(frame)

    video.release()


def creategt(path, savepath):

    #extract vidname
    vidname = path.split("/")[3]
    #extract imagename
    samplelist = os.listdir(path)
    pathlist = []
    for i in range(len(samplelist)):
        if samplelist[i].split(".")[-1] == "txt":
            pathlist.append(samplelist[i])

    #create masked array
    gt = ma.array(np.zeros((len(pathlist), 6)), mask = True, dtype = np.float64)

    for i in range(len(pathlist)):
        with open(path + pathlist[i], "r") as text:

            listtext = list(text)

            comp = []
            for j in range(len(listtext)):
                comp.append(listtext[j].split(" ")[1])

            #select bbox by selecting max centerx
            indx = comp.index(max(comp)) 

            #select relevant row
            relrow = list(map(float, listtext[indx].split(" ")))

            dimx = 1920
            dimy = 1080

            label = relrow[0]
            centerx = relrow[1] * dimx
            centery = relrow[2] * dimy
            width = relrow[3] * dimx
            height = relrow[4] * dimy

            #class
            gt[i,0] = label
            #confidence
            gt[i,1] = 1
            #coords in "cor" format
            gt[i,2] , gt[i,3] , gt[i,4] , gt[i,5] = centerx, centery, width, height
    gt[:,2:6] = h.cencor(gt[:,2:6])

    h.writemaarray(gt, savepath + "gtsynthetic/" + vidname, "gt from synthetic data")

#plot gt of left bbox (aspect ratio) [same as above but left bbox and plot instead of saving]
def creategtleft(path, savepath):

    #extract vidname
    vidname = path.split("/")[3]
    #extract imagename
    samplelist = os.listdir(path)
    pathlist = []
    for i in range(len(samplelist)):
        if samplelist[i].split(".")[-1] == "txt":
            pathlist.append(samplelist[i])

    #create masked array
    gt = ma.array(np.zeros((len(pathlist), 6)), mask = True, dtype = np.float64)

    for i in range(len(pathlist)):
        with open(path + pathlist[i], "r") as text:

            listtext = list(text)

            comp = []
            for j in range(len(listtext)):
                comp.append(listtext[j].split(" ")[1])

            #select bbox by selecting max centerx
            indx = comp.index(min(comp)) 

            #select relevant row
            relrow = list(map(float, listtext[indx].split(" ")))

            dimx = 1920
            dimy = 1080

            label = relrow[0]
            centerx = relrow[1] * dimx
            centery = relrow[2] * dimy
            width = relrow[3] * dimx
            height = relrow[4] * dimy

            #class
            gt[i,0] = label
            #confidence
            gt[i,1] = 1
            #coords in "cor" format
            gt[i,2] , gt[i,3] , gt[i,4] , gt[i,5] = centerx, centery, width, height

    gt[:,2:6] = h.cencor(gt[:,2:6])
    return gt


def createvidgt(path, videosavepathname, gt):

    #extract vidname
    vidname = path.split("/")[3]
    #extract imagename
    samplelist = os.listdir(path)
    imagelist = []
    for i in range(len(samplelist)):
        if samplelist[i].split(".")[-1] == "jpg":
            imagelist.append(samplelist[i])

    #create videowriter
    video = cv2.VideoWriter(videosavepathname + ".avi",cv2.VideoWriter_fourcc(*"XVID"),30,(1920,1080))

    for i in range(len(imagelist)):
        frame = cv2.imread(path + imagelist[i])
        if gt.mask[i, 0] == False:
            frame = cv2.rectangle(frame,(int(gt[i,0]),int(gt[i,1])),(int(gt[i,2]),int(gt[i,3])), (0,255,0),2) #gt
        video.write(frame)

    video.release()



 
#createvid(path, videosavepath)
gt = creategtleft(path,genpath)
createvidgt(path,genpath + "synthasptest",gt[:,2:6])
gtasp = h.corasp(gt[:,2:6])
plt.scatter(range(gtasp.shape[0]-1), gtasp[:-1,3])
plt.xlabel("Frame [k]")
plt.ylabel("aspect ratio")
plt.grid()
plt.tight_layout()
plt.show()





