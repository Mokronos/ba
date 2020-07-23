#pipeline
#seperate following sections --> check at every section if section already done for clip esp. at raw data step(takes long)
#get raw data
#clean data
#analytics
#algorithms
#analytics

import numpy as np
import os
import matplotlib.pyplot as plt
import helper as h
import of
import kal
import cv2
import numpy.ma as ma
textpath = "drawmaininfo1"
#load text with clip times and main clip

#get raw data
#cut clips according to timestamps from video
def main(skip):


    
    #read info about clips and define sourcevideo name from its path(-.mp4)
    mainvidpath, info = h.readtimes(textpath)
    mainvid = mainvidpath.split(".")[-2]
    mainvid = mainvid.split("/")[-1]
    datapath = "./data/"
    macropath = datapath + mainvid + "/macroanalysis/"

    if skip < 1:
        #delete all existing analysis and alorithm folders(so that if i changed name of some files the old ones get deleted)
        h.delfolders(mainvid)
        
        """
        ####################################################################
        load raw data
        ####################################################################
        """
        
           #loop over clips --> create folders for clips and cut them up (if u change clip length but not name u need to delete the folder for that clip, otherwise it will just be skipped and not updated)
        for i in range(np.shape(info)[0]):

            #define clippath to give different methods
            clippath = datapath + mainvid + "/" + info[i][2]
            clipname = h.extract(clippath)

            if not os.path.exists(clippath):

                #create folders and cut main vid
                print("creating folders and cutting: " + info[i][2])

                h.clipfolder(mainvid, info[i][2])

            if not os.path.exists(clippath + "/data/sourceinfo.txt"):
                h.cutmainvid(mainvid, info[i][0], info[i][1], info[i][2], mainvidpath)

            if not os.path.exists(clippath + "/detections/raw.txt"):
                
                #use yolo to get bbox of current clipfilename
                print("yolo running on: " + info[i][2])

                bbox = h.clipyolodet(clippath)

                print("writing detections to file...")

                h.writemaarray(bbox, clippath + "/detections/raw", "rawdetectiontestheader")
        
            #create folders
            h.makedirsx(clippath + "/analysis")
            h.makedirsx(clippath + "/algorithms")
            h.makedirsx(clippath + "/algorithms/kalman")
            h.makedirsx(clippath + "/algorithms/fm")
            h.makedirsx(clippath + "/algorithms/of")
            #create folder for pre analysis
            h.makedirsx(clippath + "/analysis/pre")
            h.makedirsx(clippath + "/analysis/post")

        #create folder to store histograms over all clips errors in
        h.makedirsx(macropath)
        h.makedirsx(macropath + "/pre")
        h.makedirsx(macropath + "/post")


        #start new loop to self label all after each other, not for every single clip(would have big pauses between clips where yolo runs)
        for i in range(np.shape(info)[0]):

            clippath = datapath + mainvid + "/" + info[i][2]
            clipname = h.extract(clippath)

            #selflabel until object is going (even partly) out of frame to focus on movement of object
            while(not os.path.exists(clippath + "/groundtruth/gt.txt")):
                gt = h.selflabel(clippath, 0)
                h.writemaarray(gt, clippath + "/groundtruth/gt" , "this is the self labeled ground truth")

        print("loaded raw data + saved detections and groundtruth")

        """
        ####################################################################
        clean data
        ####################################################################
        """

        #clean data
        for i in range(np.shape(info)[0]):

            clippath = (datapath + mainvid + "/" + info[i][2])
            clipname = h.extract(clippath)

            if not os.path.exists(clippath + "/detections/clean.txt"):
                #read gt and raw detections from text file 
                gt = h.readmaarray(clippath + "/groundtruth/gt") #need class and conf for cleanup
                bboxraw = h.readmaarray(clippath + "/detections/raw") #need class and conf for cleanup

                #clean data
                bboxclean = h.cleanup(bboxraw, 0, gt)

                #write clean data back
                h.writemaarray(bboxclean, clippath + "/detections/clean" , "cleaned up detections")
                print("cleaned data from: " + clipname)
    """
    ############################################################################
    run opical flow
    ############################################################################
    """
    
    #define windowsizes outside to better retrieve file later (if u dont want to run this section all the time)
    windows = [3,9,27]
    if skip < 2:

        #define of parameters
        lkparams = dict( winSize  = (3,3),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


        #run optical flow on clips with center as init
        for i in range(np.shape(info)[0]):


            clippath = datapath + mainvid + "/" + info[i][2]
            clipname = h.extract(clippath)

            #find good points to track and save them
            framenbr = 0
            h.drawgoodpoints(clippath, framenbr, clippath + "/algorithms/of/goodpoints")

            #run optical flow
            gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
            gtcen = h.corcen(gt)
            #define points to track
            pointinfo = np.array([["ofcenter",gtcen[0,0],gtcen[0,1]],["oftopleft",gt[0,0],gt[0,1]],["oftopright", gt[0,2],gt[0,1]]])
            for k in range(len(windows)):
                lkparams["winSize"] = (windows[k],windows[k])
                for j in range(pointinfo.shape[0]):
                    memof = of.ofcustompoint(clippath,[pointinfo[j,1],pointinfo[j,2]],gt, lkparams)
                    h.writemaarray(memof, clippath + "/algorithms/of/" + clipname + pointinfo[j,0] + "win" + str(windows[k]), "of initialized with " + pointinfo[j,0] + "of gt")
                    of.visof(memof,clippath, clippath + "/algorithms/of/" + clipname + pointinfo[j,0]+ "win" + str(windows[k]))

                #save of center and width in one txt file to better retrieve it later
                #load 3 points
                ofcenter = h.readmaarray(clippath + "/algorithms/of/" + clipname + pointinfo[0,0]+ "win" + str(windows[k]))
                oftopleft = h.readmaarray(clippath + "/algorithms/of/" + clipname + pointinfo[1,0]+ "win" + str(windows[k]))
                oftopright = h.readmaarray(clippath + "/algorithms/of/" + clipname + pointinfo[2,0]+ "win" + str(windows[k]))
                
                #put width into from 2 points into ofpoints array and later give whole array with(centerx, centery, width) to kalman
                maarray = ma.array(np.zeros((ofcenter.shape[0], 3)), mask = True, dtype = np.float64)
                maarray[:,:2] = ofcenter
                for j in range(ofcenter.shape[0]):
                    maarray[j,2] = oftopright[j,0] - oftopleft[j,0]

                ofpoints = maarray
                #save in txt as (centerx,centery, width)
                h.writemaarray(ofpoints, clippath + "/algorithms/of/" + clipname + "ofcenwidth"+ "win" + str(windows[k]), "of initialized with centerx, centery, width(top right and top left corner)  of bbox(gt)")


                print("ran optical flow on: " + clipname + " winSize: " + str(windows[k])+ "*" + str(windows[k]) )
        
        print("done with optical flow")
    """
    ############################################################################
    pre-analysis
    ############################################################################
    """

    #not being used at the moment 
    binsize = 3


    if skip < 3:


        #analysis 
        for i in range(np.shape(info)[0]):

            clippath = datapath + mainvid + "/" + info[i][2]
            clipname = h.extract(clippath)

            #over time
            #read bboxes
            bboxclean = h.readmaarray(clippath + "/detections/clean")[:,2:6]
            gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
            
            #corner representation
            #plot and save yolo bboxes
            ylabel = ["xmin [px]","ymin [px]","xmax [px]","ymax [px]"]
            #plot and save gt bboxes
            h.timeplot2(gt, bboxclean, ylabel, ["Frame [k]"]*4, clippath + "/analysis/pre/"+ clipname +"otyologt")

            #center representation
            #plot and save yolo bboxes
            ylabel = ["Mittelpunktx [px]","Mittelpunkty [px]","Breite [px]","Höhe [px]"]
            #plot and save gt bboxes
            h.timeplot2(h.corcen(gt),h.corcen(bboxclean),ylabel , ["Frame [k]"]*4, clippath + "/analysis/pre/"+ clipname +"otyologt")


            #aspect ratio representation
            #plot and save yolo bboxes
            ylabel = ["Mittelpunktx [px]","Mittelpunkty [px]","Breite [px]","Seitenverhältnis"]
            #plot and save gt bboxes
            h.timeplot2(h.corasp(gt),h.corasp(bboxclean), ylabel, ["Frame [k]"]*4, clippath + "/analysis/pre/"+ clipname +"otyologt")

            for k in range(len(windows)):


                ofcenwidth = h.readmaarray(clippath + "/algorithms/of/" + clipname + "ofcenwidth" + "win" + str(windows[k]))
                ofcenright = h.readmaarray(clippath + "/algorithms/of/" + clipname + "oftopright"+ "win" + str(windows[k]))
                ofcenleft = h.readmaarray(clippath + "/algorithms/of/" + clipname + "oftopleft"+ "win" + str(windows[k]))
                ofleftright = h.connectleftright(ofcenleft,ofcenright)
                ofpointsgt = h.createofgt(gt)

                ylabel = ["Mittelpunktx [px]","Mittelpunkty [px]","Breite [px]","Seitenverhältnis"]
                #compare ofcenter and width with gt center and width
                h.timeplot2(h.corcen(gt)[:,:3], ofcenwidth, ylabel, ["Frame [k]"]*4, clippath + "/analysis/pre/"+ clipname +"otofgt"+ "win" + str(windows[k]),["green", "black"],["Ground Truth", "Optischer Fluss"])
                
                #plot of points over time compared with gt
                ylabel = ["linksobenx [px]","linksobeny [px]","rechtsobenx [px]","rechtsobeny [px]"]
                h.timeplot2(ofpointsgt, ofleftright, ylabel, ["Frame [k]"]*4, clippath + "/analysis/pre/"+ clipname +"otofgt"+ "win" + str(windows[k]),["green", "black"],["Ground Truth", "Optischer Fluss"])


                #plot errors and rmse over time for every clip and every winsize
                ylabel = ["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Höhe Fehler [px]"]
                h.errorplot( h.corcen(gt)[:,:3], ofcenwidth, ylabel, ["Frame [k]"]*4, clippath + "/analysis/pre/" + clipname +"oferror" + "win" + str(windows[k]))
                h.rmseplot( h.corcen(gt)[:,:3] ,ofcenwidth, ylabel, ["Frame [k]"]*4,clippath + "/analysis/pre/" + clipname +"ofrmse" + "win" + str(windows[k]))
                np.savetxt(clippath + "/analysis/pre/" + clipname +"ofrmsecw" + "win" + str(windows[k])+ ".txt", (h.calcrmse(h.corcen(gt)[:,:3],ofcenwidth)) ,fmt = "%1.2f", header ="rmse for center(x,y) and width")


                ylabel = ["linksobenx RMSE [px]","linksobeny RMSE [px]","rechtsobenx RMSE [px]","rechtsobeny RMSE [px]"]
                h.errorplot( ofpointsgt, ofleftright, ylabel, ["Frame [k]"]*4, clippath + "/analysis/pre/" + clipname +"oferror" + "win" + str(windows[k]))
                h.rmseplot( ofpointsgt,ofleftright, ylabel, ["Frame [k]"]*4,clippath + "/analysis/pre/" + clipname +"ofrmse" + "win" + str(windows[k]))

                #save rmse as txt
                np.savetxt(clippath + "/analysis/pre/" + clipname +"ofrmselr" + "win" + str(windows[k])+ ".txt", (h.calcrmse(ofpointsgt,ofleftright)) ,fmt = "%1.2f", header ="rmse for topleft(x,y) and topright(x,y)")


            print("plotted over time:" + clipname)

        #plot histograms for the 3 of points and the width --> for every clip and overall (+for every window size)

        for k in range(len(windows)):
            totalerrorofcen = []
            totalerroroflr = []
            for i in range(np.shape(info)[0]):

                clippath = (datapath + mainvid + "/" + info[i][2])
                clipname = h.extract(clippath)
                bboxclean = h.readmaarray(clippath + "/detections/clean")[:,2:6]
                gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
                #paths need to be correct for point nicknames of optical flow points(pointinfo above)
                ofcenwidth = h.readmaarray(clippath + "/algorithms/of/" + clipname + "ofcenwidth" + "win" + str(windows[k]))
                gtof = h.corcen(gt)[:,:3]
                ofcenright = h.readmaarray(clippath + "/algorithms/of/" + clipname + "oftopright"+ "win" + str(windows[k]))
                ofcenleft = h.readmaarray(clippath + "/algorithms/of/" + clipname + "oftopleft"+ "win" + str(windows[k]))
                ofleftright = h.connectleftright(ofcenleft,ofcenright)
                ofpointsgt = h.createofgt(gt)

                errorof = h.error(ofcenwidth, gtof)
                totalerrorofcen = h.apperror(errorof,totalerrorofcen)
                erroroflr = h.error(ofleftright, ofpointsgt)
                totalerroroflr = h.apperror(erroroflr,totalerroroflr)
                
                #plot of error hist (only centerx and centery needed)
                ylabel = ["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Seitenverhätnis Fehler"]
                h.hist(np.array(errorof),"cen", ylabel, ["norm. Anz. an Fehlern"]*4, clippath + "/analysis/pre/"+ clipname + "ofhist" + "win" + str(windows[k]))
                ylabel = ["linksobenx [px]","linksobeny [px]","rechtsobenx [px]","rechtsobeny [px]"]
                h.hist(np.array(erroroflr),"cen", ylabel, ["norm. Anz. an Fehlern"]*4, clippath + "/analysis/pre/"+ clipname + "ofhist" + "win" + str(windows[k]))

                
            #plot overall errors as histogramms (over all clips)
            ylabel = ["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Seitenverhätnis Fehler"]
            h.hist(np.array(totalerrorofcen),"cen", ylabel, ["norm. Anz. an Fehlern"]*4, macropath + "pre/preofhist"+ "win" + str(windows[k]) )
            ylabel = ["linksobenx [px]","linksobeny [px]","rechtsobenx [px]","rechtsobeny [px]"]
            h.hist(np.array(totalerroroflr),"cen", ylabel, ["norm. Anz. an Fehlern"]*4, macropath + "pre/preofhist"+ "win" + str(windows[k]))


            #save errors to save std and mean from best performing optical flow for kalman filter (the one with highest windowsize in this case)
            if k == len(windows)-1:
                totalerrorofcenmem = totalerrorofcen




        #ploterrors over all clips

        totalerrorcor = []
        totalerrorcen = []
        totalerrorasp = []
        #loop over clips and extract bboxes and gt --> then sum up error
        for i in range(np.shape(info)[0]):
             
            clippath = (datapath + mainvid + "/" + info[i][2])
            clipname = h.extract(clippath)
            #read bboxes
            bboxcor = h.readmaarray(clippath + "/detections/clean")[:,2:6]
            gtcor = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
            bboxcen = h.corcen(bboxcor)
            gtcen = h.corcen(gtcor)
            bboxasp = h.corasp(bboxcor)
            gtasp = h.corasp(gtcor)


            errorcor = h.error(bboxcor, gtcor)
            totalerrorcor = h.apperror(errorcor,totalerrorcor)
            errorcen = h.error(bboxcen, gtcen)
            totalerrorcen = h.apperror(errorcen,totalerrorcen)
            errorasp = h.error(bboxasp, gtasp)
            totalerrorasp = h.apperror(errorasp,totalerrorasp)


        h.hist(np.array(totalerrorcor),"cor", ["xmin Fehler [px]","ymin Fehler [px]","xmax Fehler [px]","ymax Fehler [px]"],["norm. Anz. an Fehlern"]*4, macropath + "/pre/prehist")
        h.hist(np.array(totalerrorcen),"cen", ["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Höhe Fehler [px]"], ["norm. Anz. an Fehlern"]*4, macropath + "pre/prehist")
        h.hist(np.array(totalerrorasp),"asp", ["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Seitenverhältnis Fehler"], ["norm. Anz. an Fehlern"]*4, macropath + "pre/prehist")


        #save mean and std as txt 
        meancor, stdcor,n = h.stats(totalerrorcor)
        meancen, stdcen,_ = h.stats(totalerrorcen)
        meanasp, stdasp,_ = h.stats(totalerrorasp)
        meanof, stdof,nof = h.stats(totalerrorofcenmem)
        np.savetxt(macropath + "pre/prestatscor.txt", (meancor,stdcor),fmt = "%1.2f",header ="mean(first row) and std(second row) for (xmin,ymin,xmax,ymax)"+ "datapoints: " + str(n))
        np.savetxt(macropath + "pre/prestatscen.txt", (meancen,stdcen),fmt = "%1.2f",header ="mean(first row) and std(second row) for (centerx,centery,width,height)"+ "datapoints: " + str(n))
        np.savetxt(macropath + "pre/prestatsasp.txt", (meanasp,stdasp),fmt = "%1.2f",header ="mean(first row) and std(second row) for (centerx,centery,width,Seitenverhältnis)"+ "datapoints: " + str(n))
        np.savetxt(macropath + "pre/prestatsof.txt", (meanof,stdof),fmt = "%1.2f",header ="mean(first row) and std(second row) for optical flow init with gt (compared to gt center and width)"+ "datapoints: " + str(nof))

        print("error pre analysis done and histogramms saved")

        #create array for mean and std to give kalman function 4 x 2 x 4
        statsinfo = np.zeros((4,2,4))
        statsinfo[0,0,:] = meancor
        statsinfo[1,0,:] = meancen
        statsinfo[2,0,:] = meanasp
        statsinfo[3,0,:3] = meanof
        statsinfo[0,1,:] = stdcor
        statsinfo[1,1,:] = stdcen
        statsinfo[2,1,:] = stdasp
        statsinfo[3,1,:3] = stdof
        #save it to file to use it later
        np.save(macropath + "pre/statsdata.npy", statsinfo)


        #deletetextfile
        h.deletefile(macropath + "pre/preavgioucorroverview.txt")

        #loop over clips and save iou of yolo/gt as txt
        ioumem = []
        for i in range(np.shape(info)[0]):

            clippath = (datapath + mainvid + "/" + info[i][2])
            clipname = h.extract(clippath)
            bboxclean = h.readmaarray(clippath + "/detections/clean")[:,2:6]
            gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]


            iou = h.iouclip(bboxclean,gt)
            np.savetxt(clippath + "/analysis/pre/preavgiou.txt", [np.mean(iou), np.std(iou)],fmt = "%1.2f",header ="avg. iou of whole clip, mean(1strow) + std(2nd row)")
            h.writeapp(macropath + "pre/preavgiouoverview.txt", clipname +": mean " + str(np.mean(iou)) + " std " +str(np.std(iou)) + "\n")
            np.savetxt(clippath + "/analysis/pre/preiou.txt", iou,fmt = "%1.2f",header ="iou of whole clip")
            ioumem.extend(iou)


        plt.hist(ioumem)
        plt.tight_layout()
        plt.savefig(macropath + "pre/preiouspread.pdf")
        plt.close("all")

        #save avg iou and avg iou as txt
        np.savetxt(macropath + "pre/preiou.txt", ioumem,fmt = "%1.2f",header ="iou of all clips(compared at places where yolo originally detected a bbox)")
        np.savetxt(macropath + "pre/preavgiou.txt", [np.mean(ioumem),np.std(ioumem)],fmt = "%1.2f",header ="avgiou of all clips, mean + std (compared at places where yolo originally detected a bbox)")
        h.writeapp(macropath + "pre/preavgiouoverview.txt", "total" +": mean " + str(np.mean(ioumem)) + " std " +str(np.std(ioumem)) + "\n")

        print("IoU pre analysis done")
        print("pre-analysis done")

        """
        ####################################################################
        correct errors
        ####################################################################
        """
        print("correcting errors")

        statsinfo = np.load(macropath + "pre/statsdata.npy")
        for i in range(np.shape(info)[0]):

            #define clippath to give different methods
            clippath = (datapath + mainvid + "/" + info[i][2])
            clipname = h.extract(clippath)

            bboxclean = h.readmaarray(clippath + "/detections/clean")[:,2:6]
            bboxcleancorr = h.correctmean(bboxclean, statsinfo[0,0,:])
            h.writemaarray(bboxcleancorr, clippath + "/detections/cleancorr" , "cleaned up detections + corrected with mean")

        
        ### plot histograms again (with corrected bbox)
        totalerrorcor = []
        totalerrorcen = []
        totalerrorasp = []
        #loop over clips and extract bboxes and gt --> then sum up error
        for i in range(np.shape(info)[0]):
             
            clippath = datapath + mainvid + "/" + info[i][2]
            clipname = h.extract(clippath)
            #read bboxes
            bboxcor = h.readmaarray(clippath + "/detections/cleancorr")
            gtcor = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
            bboxcen = h.corcen(bboxcor)
            gtcen = h.corcen(gtcor)
            bboxasp = h.corasp(bboxcor)
            gtasp = h.corasp(gtcor)

            errorcor = h.error(bboxcor, gtcor)
            totalerrorcor = h.apperror(errorcor,totalerrorcor)
            errorcen = h.error(bboxcen, gtcen)
            totalerrorcen = h.apperror(errorcen,totalerrorcen)
            errorasp = h.error(bboxasp, gtasp)
            totalerrorasp = h.apperror(errorasp,totalerrorasp)


        h.hist(np.array(totalerrorcor),"cor", ["xmin Fehler [px]","ymin Fehler [px]","xmax Fehler [px]","ymax Fehler [px]"],["norm. Anz. an Fehlern"]*4, macropath + "/pre/prehistcorr")
        h.hist(np.array(totalerrorcen),"cen", ["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Höhe Fehler [px]"], ["norm. Anz. an Fehlern"]*4, macropath + "pre/prehistcorr")
        h.hist(np.array(totalerrorasp),"asp", ["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Seitenverhältnis Fehler"], ["norm. Anz. an Fehlern"]*4, macropath + "pre/prehistcorr")

        #deletetextfile
        h.deletefile(macropath + "pre/preavgioucorroverview.txt")

        #loop to get iou
        ioumem = []
        for i in range(np.shape(info)[0]):

            clippath = datapath + mainvid + "/" + info[i][2]
            clipname = h.extract(clippath)
            bboxcleancorr = h.readmaarray(clippath + "/detections/cleancorr")
            gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]


            iou = h.iouclip(bboxcleancorr, gt)
            np.savetxt(clippath + "/analysis/pre/preavgioucorr.txt", [np.mean(iou), np.std(iou)],fmt = "%1.2f",header ="avg. iou of whole clip, mean(1strow) + std(2nd row)")
            h.writeapp(macropath + "pre/preavgioucorroverview.txt", clipname +": mean " + str(np.mean(iou)) + " std " +str(np.std(iou)) + "\n")
            np.savetxt(clippath + "/analysis/pre/preioucorr.txt", iou,fmt = "%1.2f",header ="iou of whole clip")
            ioumem.extend(iou)


        plt.hist(ioumem)
        plt.tight_layout()
        plt.savefig(macropath + "pre/preiouspreadcorr.pdf")
        plt.close("all")

        #save avg iou and avg iou as txt
        np.savetxt(macropath + "pre/preioucorr.txt", ioumem,fmt = "%1.2f",header ="iou of all clips(compared at places where yolo originally detected a bbox)")
        np.savetxt(macropath + "pre/preavgioucorr.txt", [np.mean(ioumem),np.std(ioumem)],fmt = "%1.2f",header ="avgiou of all clips, mean + std (compared at places where yolo originally detected a bbox)")
        h.writeapp(macropath + "pre/preavgioucorroverview.txt", "total" +": mean " + str(np.mean(ioumem)) + " std " +str(np.std(ioumem)) + "\n")

        print("correcting errors done")
    """
    ####################################################################
    algorithms
    ####################################################################
    """

    """
    ####################################################################
    main kalman filter loops
    ####################################################################
    """


    #define different models for kalman filter (pair model with representation --> need representation to be able to know how to transform bboxes)
    modelinfo = np.array(["simplecen","cenof","aspof","aspofwidth","aspofwidthman"])

    if skip < 4:
        
        statsinfo = np.load(macropath + "pre/statsdata.npy")
        
        #create a bunch of folders for the different models
        for i in range(np.shape(info)[0]):

            #define clippath to give different methods
            clippath = datapath + mainvid + "/" + info[i][2]
            clipname = h.extract(clippath)

            for j in range(modelinfo.shape[0]):
                h.makedirsx(clippath + "/algorithms/kalman/" + modelinfo[j])
                h.makedirsx(clippath + "/analysis/post/" + modelinfo[j])
                h.makedirsx(macropath + "post/" + modelinfo[j])

        #use kalman filter on all clips repr: aspect ratio
        #TODO have custom line here with representations(can be 2 models for 1 representation)
        #loop over those 
        for i in range(np.shape(info)[0]):

            clippath = datapath + mainvid + "/" + info[i][2]
            clipname = h.extract(clippath)
            bboxcleancorr = h.readmaarray(clippath + "/detections/cleancorr")
            gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
            ofpoints = h.readmaarray(clippath + "/algorithms/of/" + clipname + "ofcenwidth"+ "win" + str(windows[2]))

            for j in range(modelinfo.shape[0]):

                #transform depending on what representation is used:
                result, memp, memk = kal.kalclip(bboxcleancorr, gt, modelinfo[j], clippath,statsinfo, ofpoint = ofpoints)

                h.writemaarray(result, clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + modelinfo[j] , "kalman filter results")

                h.viskal(bboxcleancorr, gt, result, clippath, clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + modelinfo[j])
                
                end = h.lastindex(gt)
                #plot p matrix and k matrix over time
                h.plotp(memp,end, clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + clipname + modelinfo[j])
                h.plotk(memk,end, clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + clipname + modelinfo[j])

            print("kalman filter ran on:" + clipname)

        print("kalman filter used on clips and results saved")

    """
    #########################################################
    post analysis
    #########################################################
    """
    if skip < 5:
        #analysis after using algorithm
        #plots over time for all representations and the results of all models each with all different parameters
        for i in range(np.shape(info)[0]):

            clippath = datapath + mainvid + "/" + info[i][2]
            clipname = h.extract(clippath)

            for j in range(modelinfo.shape[0]):
                results = h.readmaarray(clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + modelinfo[j])
                gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
                bboxcleancorr = h.readmaarray(clippath + "/detections/cleancorr")
                #corner representation
                #plot and save results 
                ylabel = ["xmin [px]","ymin [px]","xmax [px]","ymax [px]"]
                h.timeplot3(gt,bboxcleancorr,results, ylabel, ["Frame [k]"]*4, clippath + "/analysis/post/"+ modelinfo[j] +"/" + clipname + modelinfo[j] + "otkal")

                #center representation
                #plot and save results
                ylabel = ["Mittelpunktx [px]","Mittelpunkty [px]","Breite [px]","Höhe [px]"]
                h.timeplot3( h.corcen(gt),h.corcen(bboxcleancorr),h.corcen(results), ylabel, ["Frame [k]"]*4, clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] + "otkal")

                #aspect ratio representation
                #plot and save yolo bboxes
                ylabel = ["Mittelpunktx [px]","Mittelpunkty [px]","Breite [px]","Seitenverhältnis"]
                h.timeplot3(h.corasp(gt),h.corasp(bboxcleancorr), h.corasp(results),ylabel, ["Frame [k]"]*4, clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] + "otkal")


            print("post analysis time plot on:" + clipname)

        #plot errors over time
        for i in range(np.shape(info)[0]):

            clippath = datapath + mainvid + "/" + info[i][2]
            clipname = h.extract(clippath)

            for j in range(modelinfo.shape[0]):
                results = h.readmaarray(clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + modelinfo[j])
                gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
                #corner representation
                #plot error and save results 
                ylabel = ["xmin Fehler [px]","ymin Fehler [px]","xmax Fehler [px]","ymax Fehler [px]"]
                h.errorplot(gt,results, ylabel, ["Frame [k]"]*4, clippath + "/analysis/post/"+ modelinfo[j] +"/" + clipname + modelinfo[j] + "otkalerror")
                #plot rmse
                ylabel = ["xmin RMSE [px]","ymin RMSE [px]","xmax RMSE [px]","ymax RMSE [px]"]
                h.rmseplot(gt,results, ylabel, ["Frame [k]"]*4, clippath + "/analysis/post/"+ modelinfo[j] +"/" + clipname + modelinfo[j] + "otkalrmse")

                #save rmse as txt
                np.savetxt(clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] + "rmsecor.txt", (h.calcrmse(gt,results)) ,fmt = "%1.2f", header ="rmse for xmin,ymin,xmax,ymax")

                #center representation
                #plot and save results
                ylabel = ["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Höhe Fehler [px]"]
                h.errorplot( h.corcen(gt),h.corcen(results), ylabel, ["Frame [k]"]*4, clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] + "otkalerror")
                ylabel = ["Mittelpunktx RMSE [px]","Mittelpunkty RMSE [px]","Breite RMSE [px]","Höhe RMSE [px]"]
                h.rmseplot( h.corcen(gt),h.corcen(results), ylabel, ["Frame [k]"]*4, clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] + "otkalrmse")

                #save rmse as txt
                np.savetxt(clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] + "rmsecen.txt", (h.calcrmse(h.corcen(gt),h.corcen(results))) ,fmt = "%1.2f", header ="rmse for centerx,centery,width,height")

                #aspect ratio representation
                #plot and save yolo bboxes
                ylabel = ["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Seitenverhältnis Fehler"]
                h.errorplot(h.corasp(gt), h.corasp(results), ylabel, ["Frame [k]"]*4, clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] + "otkalerror")
                #plot rmse 
                ylabel = ["Mittelpunktx RMSE [px]","Mittelpunkty RMSE [px]","Breite RMSE [px]","Seitenverhältnis RMSE"]
                h.rmseplot(h.corasp(gt), h.corasp(results), ylabel, ["Frame [k]"]*4, clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] + "otkalrmse")

                #save rmse as txt
                np.savetxt(clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] + "rmseasp.txt", (h.calcrmse(h.corasp(gt),h.corasp(results))) ,fmt = "%1.2f", header ="rmse for centerx,centery,width,asp")


            print("post analysis rmse time plot on:" + clipname)


        #plot errors after algorithms
        for j in range(modelinfo.shape[0]):

            totalerrorcor = []
            totalerrorcen = []
            totalerrorasp = []
            #loop over clips and extract bboxes and gt --> then sum up error
            for i in range(np.shape(info)[0]):
                 
                clippath = (datapath + mainvid + "/" + info[i][2])
                clipname = h.extract(clippath)
                #read bboxes
                resultscor = h.readmaarray(clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + modelinfo[j])
                gtcor = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
                resultscen = h.corcen(resultscor)
                gtcen = h.corcen(gtcor)
                resultsasp = h.corasp(resultscor)
                gtasp = h.corasp(gtcor)

                errorcor = h.error(resultscor, gtcor)
                totalerrorcor = h.apperror(errorcor,totalerrorcor)
                errorcen = h.error(resultscen, gtcen)
                totalerrorcen = h.apperror(errorcen,totalerrorcen)
                errorasp = h.error(resultsasp, gtasp)
                totalerrorasp = h.apperror(errorasp,totalerrorasp)

            h.hist(np.array(totalerrorcor),"cor",["xmin Fehler [px]","ymin Fehler [px]","xmax Fehler [px]","ymax Fehler [px]"] , ["norm. Anz. an Fehlern"]*4, macropath + "post/" + modelinfo[j] + "/" + modelinfo[j] + "posthist")
            h.hist(np.array(totalerrorcen),"cen", ["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Höhe Fehler [px]"], ["norm. Anz. an Fehlern"]*4, macropath + "post/" + modelinfo[j] + "/" + modelinfo[j] + "posthist")
            h.hist(np.array(totalerrorasp),"asp",["Mittelpunktx Fehler [px]","Mittelpunkty Fehler [px]","Breite Fehler [px]","Seitenverhältnis Fehler"], ["norm. Anz. an Fehlern"]*4, macropath + "post/" + modelinfo[j] + "/" + modelinfo[j] + "posthist")

            #save mean and std as txt 
            meancor, stdcor,n = h.stats(totalerrorcor)
            meancen, stdcen,_= h.stats(totalerrorcen)
            meanasp, stdasp,_ = h.stats(totalerrorasp)
            np.savetxt(macropath + "post/" + modelinfo[j] + "/" + modelinfo[j] + "meanstdcor.txt", (meancor,stdcor),fmt = "%1.2f",header ="mean(first row) and std(second row) for (xmin,ymin,xmax,ymax)" + "datapoints: " + str(n))
            np.savetxt(macropath + "post/" + modelinfo[j] + "/" + modelinfo[j] + "meanstdcen.txt", (meancen,stdcen),fmt = "%1.2f",header ="mean(first row) and std(second row) for (centerx,centery,width,height)" + "datapoints: " + str(n))
            np.savetxt(macropath + "post/" + modelinfo[j] + "/" + modelinfo[j] + "meanstdasp.txt", (meanasp,stdasp),fmt = "%1.2f",header ="mean(first row) and std(second row) for (centerx,centery,width,aspect ratio)" + "datapoints: " + str(n))
        print("post error analysis done")

        #loop over different parameters of algorithms
        for j in range(modelinfo.shape[0]):

            #deletetextfile
            h.deletefile(macropath +"post/" + modelinfo[j] + "/" + modelinfo[j] + "avgiouoverview.txt")

            #loop over clips and save iou of yolo/gt as txt
            ioumem = []
            for i in range(np.shape(info)[0]):

                clippath = (datapath + mainvid + "/" + info[i][2])
                clipname = h.extract(clippath)
                results = h.readmaarray(clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + modelinfo[j])
                gt = h.readmaarray(clippath + "/groundtruth/gt")[:,2:6]
                bboxcleancorr = h.readmaarray(clippath + "/detections/cleancorr")[:,2:6]

                h.plotiou(gt, results, ["IoU"], ["Frame [k]"], clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] +  "iouot")

                iou = h.iouclip(results,gt)
                np.savetxt(clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] +  "avgiou.txt", [np.mean(iou),np.std(iou)],fmt = "%1.2f",header ="avg. iou of whole clip, mean + std (only when detection exists, no \"punishment\" for no detection)")
                h.writeapp(macropath +"post/" + modelinfo[j] + "/" + modelinfo[j] + "avgiouoverview.txt", clipname +": mean " + str(np.mean(iou)) + " std " +str(np.std(iou)) + "\n")
                np.savetxt(clippath + "/analysis/post/" + modelinfo[j] + "/" + clipname + modelinfo[j] + "iou.txt", iou,fmt = "%1.2f",header ="iou of whole clip")


                ioumem.extend(iou) 
            plt.hist(ioumem)
            plt.tight_layout()
            plt.savefig(macropath + "post/" + modelinfo[j] + "/" + modelinfo[j] + "iouspread.pdf")
            plt.close("all")
            np.savetxt(macropath + "post/" + modelinfo[j] + "/" + modelinfo[j] + "iou.txt", ioumem,fmt = "%1.2f",header ="iou of all clips(compared at places where yolo originally detected a bbox)")
            np.savetxt(macropath + "post/" + modelinfo[j] + "/" + modelinfo[j] + "avgiou.txt", [np.mean(ioumem), np.std(ioumem)],fmt = "%1.2f",header ="avgiou of all clips, mean + std(compared at places where yolo originally detected a bbox)")
            h.writeapp(macropath +"post/" + modelinfo[j] + "/" + modelinfo[j] + "avgiouoverview.txt", "total" +": mean " + str(np.mean(ioumem)) + " std " +str(np.std(ioumem)) + "\n")


        print("iou post analysis done")

        print("post-analysis done")

#skip: 0: from start, 1: from optical flow, 2: from pre analysis, 3: from kalman filter, 4: from post analysis
skip = 0
main(skip)


 
