import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import cv2

np.set_printoptions(threshold=sys.maxsize, suppress=True)


#for masked arrays
import numpy.ma as ma


#################################################
#helper functions:
#################################################



#make dir if it doesnt exist
def makedirsx(path):
    if not os.path.exists(path):
        os.makedirs(path)

#read info of time segments and custom clipnames
def readtimes(filename):

    with open(filename + ".txt", "r") as text:

        listtext = list(text)
        mainvidpath = listtext[0]
        mainvidpath = "".join(list(mainvidpath[:-1]))
        listtext.pop(0)
        startend = []
        for idx,line in enumerate(listtext):
            start = line.split()[0]
            end = line.split()[1]
            name = line.split()[2]
            start = start.split(":")
            end = end.split(":")
            start = [int(x) for x in start]
            end = [int(x) for x in end]
            start = start[0]*60 + start[1]
            end = end[0]*60 + end[1]
            startend.append([start,end,name])

    return mainvidpath, startend


#creates folders for 1 clip
def clipfolder(mainvidname, clipname):

    folders = ["data", "detections", "groundtruth"]
    b = "/"
    d = "./data/"
    if not os.path.exists(d + mainvidname):
        os.makedirs(d + mainvidname)
    
    if not os.path.exists(d + mainvidname + b + clipname):
        for i in range(len(folders)):

            os.makedirs(d + mainvidname + b + clipname + b + folders[i])


#returns index of first unmasked value of in array (return -99 if no unmasked entry found)
def getstartindex(array):
    for i in range(np.shape(array)[0]):
        if array.mask[i,0] == False:
            return i
    return -99

#writes masked array to .txt with masked values deleted
def writemaarray(maskedarray, filepath, header):
    
    #open text file
    with open(filepath + ".txt", "w")  as textfile:

        #add header to first line
        textfile.write(header + "\n")
        
        #loop over rows and framenumber + values TODO make values int (except the confidence)
        for i in range(np.shape(maskedarray)[0]):
            textfile.write(str(i))
            line = maskedarray[i, maskedarray[i,:].mask == False]
            for value in line:
                textfile.write(" " + str(round(value,2)))
            textfile.write("\n")

#reads masked array from text file(creates rectangular array with "shorter" rows masked)
def readmaarray(filename):
    with open(filename + ".txt", "r") as text:

        #make iterator list to use it more than once
        listtext = list(text)

        #remove header in first line
        listtext.pop(0)

        #get maxrow and maxcol(pop of first index of every line bc it is the framenumber which is not needed in array)
        maxrow = 0
        colcounter = []
        for idx,line in enumerate(listtext):
            splitline = [float(x) for x in line.split()]
            splitline.pop(0)
            colcounter.append(len(splitline))
            maxrow = idx + 1

        maxcol = max(colcounter)

        #create fully masked array with (maxrow,maxcol) shape
        maskedarray = ma.array(np.zeros((maxrow, maxcol)), mask = True, dtype = np.float64)

        #write values from text file into array(masked values get overwritten) --> excess space of shorter rows stays masked + pop off frame number again
        for idxrow,line in enumerate(listtext):
            splitline = [float(x) for x in line.split()]
            splitline.pop(0)
            for idxcol, value in enumerate(splitline):
                maskedarray[idxrow, idxcol] = value
        
        #return masked array        
        return maskedarray

#cuts main video(mainvidfile) from start(second) to end(second)(read those out of a text file)
def cutmainvid(mainvidname, start, end, clipname, sourcepath):


    #define datapath(where frames and clip goes)
    datapath = "./data/" + mainvidname + "/" + clipname + "/data"
    clippath =  datapath + "/" + clipname + ".avi"

    #load video
    video = cv2.VideoCapture(sourcepath)
    ok, frame = video.read()
    
    #get fps
    fps = video.get(cv2.CAP_PROP_FPS)

    #get start and finish frame from fps and start/end
    #multiply fps with seconds = startingframe(is float bc fps is 29.97) --> make int (rounding down)
    sframe = int(start * fps)
    eframe = int(end * fps)

    #if start 0 and end 0 given --> take whole main video as clip(mainly for synthetic data)
    if start == 0 and end == 0:
        sframe = 0
        eframe = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        end = eframe /fps

    #set starting frame for VideoCapture
    video.set(cv2.CAP_PROP_POS_FRAMES, sframe)

    #set up VideoWriter
    writefps = fps
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    dimensions = (1920, 1080)
    cutvid = cv2.VideoWriter(clippath, fourcc, writefps, dimensions) 

    #create text file with clip properties
    videoinfotext(datapath,sourcepath,fps,sframe, eframe, start, end, clippath, writefps)  

    #loop from sframe to eframe, save every single frame as clipname + frame_number and whole clip as .avi
    for i in range(sframe, eframe):


        #read current frame
        ok, frame = video.read()

        #write current frame to video
        cutvid.write(frame)

        #write current frame to file as png
        cv2.imwrite(datapath + "/" + clipname + "#" + str(i-sframe) + ".png", frame)

    #release Videocapture
    video.release()
    cutvid.release()
    cv2.destroyAllWindows()

#write some info about where the clip was cut from to ensure reproducibility
def videoinfotext(targetpath, sourcevideo, sourcefps, startframe, endframe, startsecond, endsecond, clippath, clipfps):

    #write all the given stuff nicely in a text file !!TODO: maybe find a nicer represantation to make it easier to read for another function
    with open(targetpath + "/sourceinfo.txt", "w") as textfile:

        textfile.write("sourcevideo:" + "\n")
        textfile.write("path: "+ str(sourcevideo) + "\n")
        textfile.write("fps: " + str(round(sourcefps, 2))+ "\n")
        textfile.write("cut from frame: " + str(startframe) + " to frame: " + str(endframe)+ " (endframe not included)" + "\n")
        textfile.write("cut from second: " + str(startsecond) + " to second: " + str(endsecond)+ " (approximate measure, better to use framenumber)" + "\n")

        textfile.write("\n")
        textfile.write("clip:" + "\n")
        textfile.write("path:" + str(clippath) + "\n")
        textfile.write("fps: " + str(round(clipfps, 2))+ "\n")
        textfile.write("total frames:"  + str(endframe-startframe) + "\n")

#returns total frames of clip given a clipname
#TODO change method of getting frames or giving clipname parameter, its ugly like this (need to carry clipname through methods that dont even need it)
def gettotalframes(clippath):

    #get clipname
    clipname = extract(clippath)

    #extract total framenumber
    vid = cv2.VideoCapture(clippath + "/data/" +clipname + ".avi")
    number = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    return number

#get yolo detections of clip and write to file 
def clipyolodet(clippath):

    clipname = extract(clippath)

    #import only in this method(import takes long) --> so only import when needed
    sys.path.insert(1,"./yolo/")
    from yolosingleimage import yolodet

    #get max number of frames
    totframes = gettotalframes(clippath)
    


    #create masked array with shape(totframes, 6*100) 100 for 100 max detections per frame --> cut off right side of array later when max detection amount of clip is known
    bbox = ma.array(np.zeros((totframes, 600)), mask = True, dtype = np.float64)

    #loop over frames of clip
    for i in range(totframes):
        print("yolo running on frame:" +str(i))
        det = yolodet(clippath + "/data/"+ clipname + "#" + str(i) + ".png")
        if det is not []:

            bbox[i,:len(det)] = det


    bbox = bbox[:, ~np.all(bbox.mask, axis=0)]



    return bbox

#extract clipname from clippath
def extract(clippath):
    
    return clippath.split("/")[-1]

#prompts user to label a clip (draw rectangle on given label)
def selflabel(clippath, label):

    #define how many detections there are per frame TODO make dependent on current frame (hotkey for next frame and option to select second box for current frame + maybe define class when labeling)
    detnmb = 1

    clipname = extract(clippath)

    #get max number of frames
    totframes = gettotalframes(clippath)

    #create masked array with shape(totalframes, (1+1+4))
    groundtruth = ma.array(np.zeros((totframes, detnmb * 6)), mask = True, dtype = np.float64)

    #loop over frames and select bboxes with cv2.selectROI() --> save them as array
    for i in range(totframes):

        #read current frame
        frame = cv2.imread(clippath + "/data/" + clipname + "#" + str(i) + ".png")

        #draw info on frame(framenumber)
        frame = cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)

        #draw bbox of last frame as reference and frame number
        if i > 0:
            if groundtruth[i-1,0] is not ma.masked:

                frame = cv2.rectangle(frame, (int(groundtruth[i-1,2]),int(groundtruth[i-1,3])), (int(groundtruth[i-1,4]),int(groundtruth[i-1,5])), (0,255,0), 1)

        #select bbox
        bbox = cv2.selectROI(frame)

        #convert bbox coord from (xmin,ymin,width,height) to (xmin,ymin,xmax,ymax)
        bbox = list(bbox)
    
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]

        #fill array with values(save bbox values) (label,confidence{100% when labeling},xmin,ymin,xmax,ymax)
        groundtruth[i,:] = [label,1] + bbox

        #remask non-detections(selectROI() returns (0,0,0,0) when not selecting any bbox --> need to remask 0s)
        if bbox == [0,0,0,0]:
            groundtruth[i,:] = ma.masked
    
    return groundtruth

#removes unwanted detections(multiples of 1 class on 1 object in 1 frame, 2nd class, detectins on 2nd object other than main object{self labeled})
def cleanup(bbox, mainlabel, groundtruth):

    #mask unwanted class e.g. if main label is 0 --> 1 is unwanted --> remove 1
    for i in range(np.shape(bbox)[0]):
        
        #loop over blocks of 6
        for j in range(np.shape(bbox)[1]//6):

            #check if entry is the wrong class and is not masked(valid)
            if bbox.mask[i,j*6] == False:
                if int(bbox[i, j*6]) != mainlabel:

                    #mask wrong class
                    bbox[i, j*6:j*6+6] = ma.masked
    
    #squish array back together --> entries are all to left
    bbox = squisharray(bbox)
    #print("unwanted class removed")
    #print(repr(bbox))

    #remove multiple detections of main class(with self labeling as position reference)
    #first check if 2 or more bboxes have overlapping area(IoU does the job) --> delete the ones with least confidence
    #then keep the one with least distance to ground truth
    for k in range(np.shape(bbox)[0]):
        #work one row at a time
        rowdata = bbox[k,bbox.mask[k,:] == False]
        #print(str(k) +"th row:")
        #print(repr(rowdata))
        #condition for breaking the loop
        while(1):
            bc = 0
            #work with one row at a time --> delete overlappings first 
            rowdata = rowdata[rowdata.mask[:] == False]
            entries = np.shape(rowdata)[0]//6
            #loop over entries
            for i in range(entries):
                #loop over other entries 
                for j in range(i+1,entries):
                    #check if entries are overlapping
                    if iou(rowdata[i*6+2:i*6+6], rowdata[j*6+2:j*6+6]) > 0:
                        #if they are overlapping mask the one with the lesser confidence
                        if rowdata[i*6+1] < rowdata[j*6+1]:
                            rowdata[i*6:i*6+6] = ma.masked
                        else:
                            rowdata[j*6:j*6+6] = ma.masked
                        #break out of loop bc array changed --> loops would try to pull already empty
                        bc = 1
                        break
                if bc == 1:
                    break
            break

        #print("overlapping removed:")
        #print(repr(rowdata))
        rowdata = rowdata[rowdata.mask[:] == False]
        entries = np.shape(rowdata)[0]//6
        #print(repr(rowdata))

        #all overlaying bboxes removed --> now pick the bbox closest to groundtruth and remove others (others are probably other objects)
        #loop over left over bbox of current row
        gtcenter = center(groundtruth[k,2:6])
        gtwh = widthheight(groundtruth[k, 2:6])
        mem = []
        for i in range(entries):
            mem.append(norm(gtcenter, center(rowdata[i*6+2:i*6+6])))
        try:
            maxpos = mem.index(min(mem))
            #make rowdata the bbox thats closest to gt
            rowdata = rowdata[maxpos*6:maxpos*6+6]
        except (ValueError, TypeError):
            pass
        
        #mask remaining bbox if it is further away from gt than two times the width/height average of the gt --> it most likely is another object at that point --> so pretty much no detection on main object 
        try:
            if norm(gtcenter, center(rowdata[2:6])) > (gtwh[0] + gtwh[1]):
                rowdata[:] = ma.masked
        except (IndexError):
            pass

        rowdata = rowdata[rowdata.mask[:] == False]
        #put final rowdata back into main bbox array
        #mask whole row
        bbox[k, :] = ma.masked
        for i in range(np.shape(rowdata)[0]):
            bbox[k,i] = rowdata[i]

    bbox = squisharray(bbox)
    return bbox

#calculate euclidean distance
def norm(center1, center2):
    
    difference = np.subtract(center1,center2)
    distance = (difference[0]**2 + difference[1]**2)**0.5

    return distance

#calculate width and height from bbox
def widthheight(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return [width, height]

#calculate coordinates of center from (xmin,ymin,xmax,ymax)
def center(bbox):

    centerx = (bbox[0] + bbox[2]) / 2
    centery = (bbox[1] + bbox[3]) / 2

    return [centerx, centery]

#calculates intersection over union for 2 given bboxes in format(xmin,ymin,xmax,ymax)
def iou(bbox1, bbox2):
    bbox1 = [int(x) for x in bbox1]
    bbox2 = [int(x) for x in bbox2]
    w_inter = min(bbox1[2],bbox2[2])  - max(bbox1[0],bbox2[0])
    h_inter = min(bbox1[3],bbox2[3])  - max(bbox1[1],bbox2[1])
    if w_inter <= 0 or h_inter <= 0:
        return 0

    I = w_inter * h_inter
    U = (bbox1[2] - bbox1[0]) * (bbox1[3]- bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - I

    
    return I/U

#moves entrys of masked array to the left if there is space(masked entries) and deletes space on the right afterwards
#make sure given array does actually have a mask --> error otherwise
def squisharray(marr):
    
    #check max needed columns of array
    c = []
    for i in range(np.shape(marr)[0]):
        c.append(marr[i,:].count())

    maxcol = max(c)

    #create new array with maxcol
    newmarr = ma.array(np.zeros((np.shape(marr)[0], maxcol)), mask = True, dtype = np.float64)

    #loop over rows to fill new array
    for i in range(np.shape(marr)[0]):

        rowdata = marr[i,marr.mask[i,:] == False]
        newmarr[i,:np.shape(rowdata)[0]] = rowdata 

    return newmarr

#compare gt with cleaned detection and return error (input coordinates --> output masked array with errors instead of coordinates) 
def error(bbox, gt):
    for i in range(np.shape(bbox)[0]):
        #if bbox at frame i is masked --> no detection there to calculate error from --> mask error of that frame
        if bbox[i,0] is not ma.masked:
            bbox[i,:] = np.subtract(bbox[i,:],gt[i,:])
        else:
            bbox[i,:] = ma.masked

    err = bbox
    return err
        
#plot histogram for errors(nx5 list with error for 4 dimensions {n = frames})
#representation is parameter for kalman filter representation with aspect ratio --> need different y and x axis for graph
#TODO improve legend
def hist(error, representation, binsize,title, subtitles, xlabel, ylabel, savepath):

    fig, axs = plt.subplots(2,2)
    fig.suptitle(title)
    axs = axs.flatten()
    membins = []
    memrange = []
    
    #figure out max range needed so that all 4 plots have same range and include all values --> make multiple of binsize so that it looks clean
    for i in range(np.shape(error)[1]):
        membins.append(abs(max(error[:,i]) - min(error[:,i])))
        memrange.append(max(abs(error[:,i])))
    maxbins = int(max(membins))
    maxrange = int(max(memrange)//binsize * binsize + binsize)
    maxbins = maxrange*2//binsize
    
    #define things for gauss function
    t = np.linspace(-maxrange,maxrange,maxrange*2 +1)
    mean, std = stats(error)

    #plot
    for i in range(np.shape(error)[1]):
        axs[i].hist(error[:,i],bins = maxbins,range = (-maxrange,maxrange), normed = True) 
        axs[i].set(title =  subtitles[i])
        axs[i].set(xlabel = xlabel[i], ylabel = ylabel[i])
        #axs[i].set(yticks = np.linspace(0,0.05,6))
        axs[i].set(xticks = range(-maxrange,maxrange+1,binsize*4))
        #plt gaussian over hist
        axs[i].plot(t, mlab.normpdf(t, mean[i], std[i]))

    if representation == "asp":
        t = np.linspace(-1,1,100)
        axs[3].clear()
        axs[3].hist(error[:,3],bins = maxbins,range = (-1,1), normed = True) 
        axs[3].set(title =  subtitles[3])
        axs[3].set(xlabel = "aspect ratio", ylabel = ylabel[3])
        #axs[3].set(xticks = np.linspace(-1,1,5))
        #axs[3].set(xlim = [-1,1])
        axs[3].plot(t, mlab.normpdf(t, mean[3], std[3]))


    plt.tight_layout()
    plt.savefig(savepath + ".pdf")
    plt.savefig(savepath + ".png")
     
#takes masked array with errors and total error (default = empty) and appends it --> returns mxn list with errors TODO if conf error is neeeded --> dont delete 2nd column
def apperror(error, toterror):


    error = error[~error.mask.any(axis=1)]

    error = error[:,-4:]

    for i in range(np.shape(error)[0]):

        toterror.append(error[i,:].tolist())
    
    return toterror
    
    
    
#plot groundtruth and detections over time(frames) --> input: xmin,ymin,xmax,ymax (masked array)
#TODO make axis scales better depending on how it works with many frames
def timeplot(bbox,title, ylim, subtitles, xlabel, ylabel, savepath):
    fig, axs = plt.subplots(2,2)
    fig.suptitle(title)
    axs = axs.flatten()
    for i in range(np.shape(bbox)[1]):
        axs[i].scatter(range(np.shape(bbox)[0]), bbox[:,i])
        axs[i].set(ylim = ylim[i])
        axs[i].set(title = subtitles[i])
        axs[i].set(xlabel = xlabel[i], ylabel = ylabel[i])
    plt.tight_layout()
    plt.savefig(savepath + ".pdf")
    plt.savefig(savepath + ".png")
        

#calc std and mean of given array(input = n x 4 array of errors)
def stats(array):
    mean = np.mean(array, axis = 0)
    std = np.std(array, axis = 0)
    return mean, std

#iou average over clip (only when detection exists)
def iouclip(bbox, gt):
    ioumem = []
    for i in range(bbox.shape[0]):
        if bbox.mask[i,0] == False and gt.mask[i,0] == False:
            ioumem.append(iou(bbox[i,:],gt[i,:]))
    return np.mean(ioumem)

#transform array from corners(xmin,ymin,xmax,ymax) to aspect(centerx,centery,width,aspect ratio) representation
def corasp(array):

    array = array.copy()

    #transform to center repr. first so its easier
    array = corcen(array)
    
    for i in range(array.shape[0]):
        if array.mask[i,0] == False:
            ar = array[i,2]/array[i,3]
            array[i, 3] = ar
            
    return array

#transform array from corners(xmin,ymin,xmax,ymax) to center(centerx,centery,width,height) representation
def corcen(array):

    array = array.copy()

    for i in range(array.shape[0]):
        if array.mask[i,0] == False:
            centerx = (array[i,2] + array[i,0])/2
            centery = (array[i,3] + array[i,1])/2
            width = array[i,2] - array[i,0]
            height = array[i,3] - array[i,1]
            p = [centerx,centery,width,height]
            for j in range(len(p)):
                array[i,j] = p[j]
             
    return array

#transform array from center(centerx,centery,width,height)  to corners(xmin,ymin,xmax,ymax) representation
def cencor(array):
    
    array = array.copy()

    for i in range(array.shape[0]):
        if array.mask[i,0] == False:
            xmin = array[i,0] - array[i,2]/2
            ymin = array[i,1] - array[i,3]/2
            xmax = array[i,0] + array[i,2]/2
            ymax = array[i,1] + array[i,3]/2
            p = [xmin,ymin,xmax,ymax]
            for j in range(len(p)):
                array[i,j] = p[j]

    return array

#transform array from aspect(centerx,centery,width,aspect ratio) to corners(xmin,ymin,xmax,ymax) representation
def aspcor(array):
    
    array = array.copy()

    for i in range(array.shape[0]):
        if array.mask[i,0] == False:
            height = array[i,2]/array[i,3]
            array[i,3] = height

    return cencor(array)

def viskal(bbox, gt, kal,clippath, savepath):

    video = cv2.VideoWriter(savepath + ".avi",cv2.VideoWriter_fourcc(*"MJPG"),2,(1920,1080))
    for i in range(bbox.shape[0]):
        img = cv2.imread(clippath + "/data/"+ extract(clippath) + "#" + str(i) + ".png",1)
        if gt.mask[i, 0] == False:
            img = cv2.rectangle(img,(int(gt[i,0]),int(gt[i,1])),(int(gt[i,2]),int(gt[i,3])), (0,255,0),2) #gt
        if bbox.mask[i, 0] == False:
            img = cv2.rectangle(img,(int(bbox[i,0]),int(bbox[i,1])),(int(bbox[i,2]),int(bbox[i,3])), (255,255,0),2) #yolo
        if kal.mask[i,0] == False:
            img = cv2.rectangle(img,(int(kal[i,0]),int(kal[i,1])),(int(kal[i,2]),int(kal[i,3])), (255,0,255),3) #kalman
        cv2.putText(img, "Ground Truth", (10,700),  cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),4)
        cv2.putText(img, "YOLOv3", (10,800),  cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0),4)
        cv2.putText(img, "Kalman", (10,900),  cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255),4)
        video.write(img)
    video.release()

