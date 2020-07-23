import cv2
import numpy.ma as ma
import helper as h
import numpy as np

#one of update
def ofupdate(frame1, frame2, pnow, lkparams):
    
    oldpoints = np.array([[[pnow]]], dtype = np.float32)
    #newpoints = ofupdate(oldframe, nowframe, oldpoints, lkparams)
    pnext, st, err = cv2.calcOpticalFlowPyrLK(frame1,frame2, oldpoints[0,:,:,:],None, **lkparams)

    return pnext, st



def ofclip(clippath, lkparams):

    #load gt for init
    gt = h.readmaarray(clippath + "/groundtruth/gt")
    gt = h.corcen(gt[:,2:6])

    #new masked array to remember of predictions
    memof = ma.array(np.zeros((np.shape(gt)[0], 2)), mask = True, dtype = np.float32)
    memof[0,:] = gt[0,:2]

    for i in range(1,gt.shape[0]):

        oldframe = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i-1) + ".png",1)
        nowframe = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i) + ".png",1)
        mem_of = np.zeros((3,2,1,2), dtype = np.float32)
        mem_of[0,:,:,:] = np.array([[(3,4)],[(1,2)]])

        oldpoints = np.array([[[memof[i-1,:]]]], dtype = np.float32)
        newpoints = ofupdate(oldframe, nowframe, memof[i-1,:], lkparams)
        #newpoints, st, err = cv2.calcOpticalFlowPyrLK(oldframe,nowframe, oldpoints[0,:,:,:],None, **lkparams)

        memof[i,:] = newpoints


    return memof

    
def visof(points,clippath, savepath):

    video = cv2.VideoWriter(savepath + ".avi",cv2.VideoWriter_fourcc(*"MJPG"),2,(1920,1080))
    for i in range(points.shape[0]):
        img = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i) + ".png",1)
        if points.mask[i, 0] == False:
            img = cv2.circle(img, tuple(map(int,(points[i]))), 2, (255,0,255), 10)
        cv2.putText(img, "Ground Truth", (10,700),  cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255),4)
        video.write(img)
    video.release()

#tracks one point through whole clip, input point as [x,y]
def ofcustompoint(clippath, point,gt, lkparams):

    #new masked array to remember of predictions
    memof = ma.array(np.zeros((np.shape(gt)[0], 2)), mask = True, dtype = np.float32)
    memof[0,:] = point


    for i in range(1,gt.shape[0]):

        oldframe = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i-1) + ".png",1)
        nowframe = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i) + ".png",1)

        oldpoints = np.array([[[memof[i-1,:]]]], dtype = np.float32)
        newpoints,status = ofupdate(oldframe, nowframe, memof[i-1,:], lkparams)
        #newpoints, st, err = cv2.calcOpticalFlowPyrLK(oldframe,nowframe, oldpoints[0,:,:,:],None, **lkparams)
        if status == 0:
            memof[i:,:] = memof[i-1,:]
            return memof
        memof[i,:] = newpoints
        

    return memof


def ofclipcustom(clippath, lkparams):

    #load gt for init
    gt = h.readmaarray(clippath + "/groundtruth/gt")
    gtcor = gt[:,2:6]
    gtcen = h.corcen(gt[:,2:6])

    #new masked array to remember of predictions
    memof = ma.array(np.zeros((np.shape(gt)[0], 4, 2)), mask = True, dtype = np.float32)
    memof[0,0,:] = gtcor[0,:2]
    memof[0,1,:] = [gtcor[0,2]-5, gtcor[0,1]] 
    memof[0,2,:] = [gtcor[0,0], gtcen[0,1]]
    memof[0,3,:] = gtcen[0,:2]

    memstatus = np.zeros((np.shape(gt)[0],4))

    for i in range(1,gt.shape[0]):

        oldframe = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i-1) + ".png",1)
        nowframe = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i) + ".png",1)

        for j in range(memof.shape[1]):

            newpoints, status = ofupdate(oldframe, nowframe, memof[i-1,j,:], lkparams)
            #newpoints, st, err = cv2.calcOpticalFlowPyrLK(oldframe,nowframe, oldpoints[0,:,:,:],None, **lkparams)
            
            memof[i,j,:] = newpoints
            memstatus[i,j] = status

            if status == 0:
                memof.mask[i,j,:] = True


    return memof, memstatus

def visofcustom(points,clippath, savepath):

    video = cv2.VideoWriter(savepath + ".avi",cv2.VideoWriter_fourcc(*"MJPG"),2,(1920,1080))
    for i in range(points.shape[0]):
        img = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i) + ".png",1)
        for j in range(points.shape[1]):
            if points.mask[i,j, 0] == False:
                img = cv2.circle(img, tuple(map(int,(points[i,j]))), 2, (255,0,255), 3)
        cv2.putText(img, "optical Flow", (10,700),  cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255),4)
        cv2.putText(img, "Frame: " + str(i), (10,100),  cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),4)
        video.write(img)
    video.release()

if __name__ == "__main__":

   
    lkparams = dict( winSize  = (3,3),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    mainvid = "Krefeld_HBF_Duisburg Ruhrort"
    clipname = "leftcurve"
    clippath = "./data/" + mainvid + "/" + clipname 
    savepath = "./gen/"
    
    #memof,status = ofclipcustom(clippath, lkparams)
    #for i in range(memof.shape[1]):
    #    h.writemaarray(memof[:,i,:], savepath + "ooooooooooooooof" + str(i) , "groundtruthheader")
    #    np.savetxt(savepath + "ooooooooooooooofstatus" + str(i) + ".txt", status[:,i])
    #visofcustom(memof, clippath, savepath + "customoftest" + str(3) + clipname)
    #h.drawgoodpoints(clippath, 17, savepath + clipname + "goodpoints")
    #i = 62
    #oldframe = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i-1) + ".png",1)
    #nowframe = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i) + ".png",1)
    #newpoints, status = ofupdate(oldframe, nowframe, [994.3,222.57], lkparams)
    #print(newpoints)
    #print(status)
    #i = 63
    #oldframe = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i-1) + ".png",1)
    #nowframe = cv2.imread(clippath + "/data/"+ h.extract(clippath) + "#" + str(i) + ".png",1)
    #newpoints, status = ofupdate(oldframe, nowframe, newpoints[0,0,:], lkparams)
    #print(newpoints)
    #print(status)
    #gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
    
    #memof = ofcustompoint(clippath, h.corcen(gt)[0,:2],gt, lkparams)
    #print(memof)
