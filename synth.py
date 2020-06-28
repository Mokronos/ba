import cv2
import os
import numpy.ma as ma
import numpy as np
import helper as h

#read images from folder --> create video --> save in right folder
#path to synthetic data (images)
path = "./sourcedata/synthetic/Noise_Moderate_Normal/Normal/"
videosavepath = "./sourcedata/sourcevideos/"
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

            xmin = centerx - width
            ymin = centery - height
            xmax = centerx + width
            ymax = centery + height

            #class
            gt[i,0] = label
            #confidence
            gt[i,1] = 1
            #coords in "cor" format
            gt[i,2] , gt[i,3] , gt[i,4] , gt[i,5] = xmin, ymin, xmax, ymax

    h.writemaarray(gt, savepath + "gtsynthetic/" + vidname, "gt from synthetic data")

createvid(path, videosavepath)
creategt(path, videosavepath)




