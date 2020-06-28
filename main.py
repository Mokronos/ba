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
import kal
textpath = "drawmaininfo1"
#load text with clip times and main clip

#get raw data
#cut clips according to timestamps from video
def main():


    
    #read info about clips and define sourcevideo name from its path(-.mp4)
    mainvidpath, info = h.readtimes(textpath)
    mainvid = mainvidpath.split(".")[-2]
    mainvid = mainvid.split("/")[-1]
    datapath = "./data/"
    macropath = datapath + mainvid + "/macroanalysis/"
    
    """
    ####################################################################
    load raw data
    ####################################################################
    """
    
       #loop over clips --> create folders for clips and cut them up (if u change clip length but not name u need to delete the folder for that clip, otherwise it will just be skipped and not updated)
    for i in range(np.shape(info)[0]):

        #define clippath to give different methods
        clippath = (datapath + mainvid + "/" + info[i][2])

        if not os.path.exists(clippath):

            #create folders and cut main vid
            print("creating folders and cutting: " + info[i][2])

            h.clipfolder(mainvid, info[i][2])
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

        clippath = (datapath + mainvid + "/" + info[i][2])

        #selflabel until object is going (even partly) out of frame to focus on movement of object
        while(not os.path.exists(clippath + "/groundtruth/gt.txt")):
            gt = h.selflabel(clippath, 0)
            h.writemaarray(gt, clippath + "/groundtruth/gt" , "groundtruthheader")

    """
    ####################################################################
    clean data
    ####################################################################
    """

    #clean data
    for i in range(np.shape(info)[0]):

        clippath = (datapath + mainvid + "/" + info[i][2])

        if not os.path.exists(clippath + "/detections/clean.txt"):
            #read gt and raw detections from text file 
            gt = h.readmaarray(clippath + "/groundtruth/gt")
            bboxraw = h.readmaarray(clippath + "/detections/raw")

            #clean data
            bboxclean = h.cleanup(bboxraw, 0, gt)

            #write clean data back
            h.writemaarray(bboxclean, clippath + "/detections/clean" , "leaned up detections")
            print("cleaned data from: " + h.extract(clippath))

    """
    ############################################################################
    pre-analysis
    ############################################################################
    """
    #define parameters for plots
    binsize = 10


    #analysis 
    for i in range(np.shape(info)[0]):

        clippath = (datapath + mainvid + "/" + info[i][2])

        #over time
        #read bboxes
        bboxclean = h.readmaarray(clippath + "/detections/clean")
        gt = h.readmaarray(clippath + "/groundtruth/gt")

        #corner representation
        #plot and save yolo bboxes
        h.timeplot(bboxclean[:,2:6], info[i][2] + "-yolo", [[0,1920],[0,1080],[0,1920],[0,1080]], ["xmin","ymin","xmax","ymax"], ["Frame"]*4, ["Pixel"]*4, clippath + "/analysis/pre/overtimeyolocor")

        #plot and save gt bboxes
        h.timeplot(gt[:,2:6], info[i][2] + "-gt", [[0,1920],[0,1080],[0,1920],[0,1080]], ["xmin","ymin","xmax","ymax"], ["Frame"]*4, ["Pixel"]*4, clippath + "/analysis/pre/overtimegtcor")
        

        #center representation
        #plot and save yolo bboxes
        h.timeplot(h.corcen(bboxclean[:,2:6]), info[i][2] + "-yolo", [[0,1920],[0,1080],[0,1920],[0,1080]], ["centerx","centery","width","height"], ["Frame"]*4, ["Pixel"]*4, clippath + "/analysis/pre/overtimeyolocen")

        #plot and save gt bboxes
        h.timeplot(h.corcen(gt[:,2:6]), info[i][2] + "-gt", [[0,1920],[0,1080],[0,1920],[0,1080]],["centerx","centery","width","height"] , ["Frame"]*4, ["Pixel"]*4, clippath + "/analysis/pre/overtimegtcen")


        #aspect ratio representation
        #plot and save yolo bboxes
        h.timeplot(h.corasp(bboxclean[:,2:6]), info[i][2] + "-yolo", [[0,1920],[0,1080],[0,1080],[0,2]], ["centerx","centery","width","aspect ratio"], ["Frame"]*4, ["Pixel"]*4, clippath + "/analysis/pre/overtimeyoloasp")

        #plot and save gt bboxes
        h.timeplot(h.corasp(gt[:,2:6]), info[i][2] + "-gt", [[0,1920],[0,1080],[0,1080],[0,2]], ["centerx","centery","width","aspect ratio"], ["Frame"]*4, ["Pixel"]*4, clippath + "/analysis/pre/overtimegtasp")


        plt.close("all")

    #ploterrors over all clips

    totalerrorcor = []
    totalerrorcen = []
    totalerrorasp = []
    #loop over clips and extract bboxes and gt --> then sum up error
    for i in range(np.shape(info)[0]):
         
        clippath = (datapath + mainvid + "/" + info[i][2])
        #read bboxes
        bboxcor = h.readmaarray(clippath + "/detections/clean")
        gtcor = h.readmaarray(clippath + "/groundtruth/gt")
        bboxcen = h.corcen(bboxcor[:,2:6])
        gtcen = h.corcen(gtcor[:,2:6])
        bboxasp = h.corasp(bboxcor[:,2:6])
        gtasp = h.corasp(gtcor[:,2:6])

        errorcor = h.error(bboxcor[:,2:6], gtcor[:,2:6])
        totalerrorcor = h.apperror(errorcor,totalerrorcor)
        errorcen = h.error(bboxcen, gtcen)
        totalerrorcen = h.apperror(errorcen,totalerrorcen)
        errorasp = h.error(bboxasp, gtasp)
        totalerrorasp = h.apperror(errorasp,totalerrorasp)



    h.hist(np.array(totalerrorcor),"cor",  binsize, "error", ["xmin","ymin","xmax","ymax"], ["Pixel"]*4, ["norm. Anz. an Fehlern"]*4, macropath + "/pre/errorscor")
    h.hist(np.array(totalerrorcen),"cen", binsize, "error", ["centerx","centery","width","height"], ["Pixel"]*4, ["norm. Anz. an Fehlern"]*4, macropath + "pre/errorscen")
    h.hist(np.array(totalerrorasp),"asp", binsize, "error", ["centerx","centery","width","aspect ratio"], ["Pixel"]*4, ["norm. Anz. an Fehlern"]*4, macropath + "pre/errorsasp")

    #save mean and std as txt 
    meancor, stdcor = h.stats(totalerrorcor)
    meancen, stdcen = h.stats(totalerrorcen)
    meanasp, stdasp = h.stats(totalerrorasp)
    np.savetxt(macropath + "pre/meanstdcor.txt", (meancor,stdcor),fmt = "%1.2f",header ="mean(first row) and std(second row) for (xmin,ymin,xmax,ymax)")
    np.savetxt(macropath + "pre/meanstdcen.txt", (meancen,stdcen),fmt = "%1.2f",header ="mean(first row) and std(second row) for (centerx,centery,width,height)")
    np.savetxt(macropath + "pre/meanstdasp.txt", (meanasp,stdasp),fmt = "%1.2f",header ="mean(first row) and std(second row) for (centerx,centery,width,aspect ratio)")

    #loop over clips and save iou of yolo/gt as txt
    for i in range(np.shape(info)[0]):

        clippath = (datapath + mainvid + "/" + info[i][2])
        bboxclean = h.readmaarray(clippath + "/detections/clean")
        gt = h.readmaarray(clippath + "/groundtruth/gt")


        iou = h.iouclip(bboxclean[:,2:6],gt[:,2:6])
        np.savetxt(clippath + "/analysis/pre/avgiou.txt", [iou],fmt = "%1.2f",header ="avg. iou of whole clip(only when detection exists, no \"punishment\" for no detection)")

    print("pre-analysis done")

    """
    ####################################################################
    algorithms
    ####################################################################
    """

    #define different models for kalman filter (pair model with representation --> need representation to be able to know how to transform bboxes)
    modelinfo = np.array(["cor", "cen", "asp"])
    #array with desccription + parameters to loop over(need name bc files need to be referenced to read later and have names for analysis)
    param = np.array([["highR", 10000,500,1],["highQ", 500, 10000, 1],["balanced", 1000,1000,1]], dtype = object)

    #create a bunch of folders for the different models and parameters
    for i in range(np.shape(info)[0]):

        #define clippath to give different methods
        clippath = (datapath + mainvid + "/" + info[i][2])

        for j in range(modelinfo.shape[0]):
            h.makedirsx(clippath + "/algorithms/kalman/" + modelinfo[j])
            h.makedirsx(clippath + "/analysis/post/" + modelinfo[j])
            h.makedirsx(macropath + "post/" + modelinfo[j])
            for k in range(param.shape[0]):
                h.makedirsx(clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + param[k, 0])
                h.makedirsx(clippath + "/analysis/post/" + modelinfo[j] + "/" + param[k, 0])
                h.makedirsx(macropath + "post/" + modelinfo[j] + "/" + param[k, 0])


    #use kalman filter on all clips repr: aspect ratio
    #TODO have custom line here with representations(can be 2 models for 1 representation)
    #loop over those 
    for i in range(np.shape(info)[0]):

        clippath = (datapath + mainvid + "/" + info[i][2])
        bboxclean = h.readmaarray(clippath + "/detections/clean")
        gt = h.readmaarray(clippath + "/groundtruth/gt")

        for j in range(modelinfo.shape[0]):
            for k in range(param.shape[0]):
            

                #transform depending on what representation is used:
                if j == 0 :
                    result = kal.kalclip(bboxclean[:,2:6], gt[:,2:6] , modelinfo[j] , param[k,1], param[k,2])

                elif j == 1:
                    result = kal.kalclip(h.corcen(bboxclean[:,2:6]), h.corcen(gt[:,2:6]) , modelinfo[j] , param[k,1], param[k,2])
                    result = h.cencor(result)

                elif j == 2:
                    result = kal.kalclip(h.corasp(bboxclean[:,2:6]), h.corasp(gt[:,2:6]) , modelinfo[j] , param[k,1], param[k,2])
                    result = h.aspcor(result)


                h.writemaarray(result, clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + param[k,0] + "/" + modelinfo[j] + param[k,0] , "kalman filter results")

                h.viskal(bboxclean[:,2:6], gt[:,2:6], result, clippath, clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + param[k,0] + "/" + modelinfo[j] + param[k,0])
    print("kalman filter used on clips and results saved")

    """
    #########################################################
    post analysis
    #########################################################
    """

    #analysis after using algorithm
    #plots over time for all representations and the results of all models each with all different parameters
    for i in range(np.shape(info)[0]):

        clippath = (datapath + mainvid + "/" + info[i][2])

        for j in range(modelinfo.shape[0]):
            for k in range(param.shape[0]):
                results = h.readmaarray(clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + param[k,0] + "/" + modelinfo[j] + param[k,0])
                gt = h.readmaarray(clippath + "/groundtruth/gt")
                #corner representation
                #plot and save results 
                h.timeplot(results, info[i][2] + "-results", [[0,1920],[0,1080],[0,1920],[0,1080]], ["xmin","ymin","xmax","ymax"], ["Frame"]*4, ["Pixel"]*4, clippath + "/analysis/post/" + modelinfo[j] + "/" + param[k,0] + "/overtimecor")

                #center representation
                #plot and save results
                h.timeplot(h.corcen(results), info[i][2] + "-results", [[0,1920],[0,1080],[0,1920],[0,1080]], ["centerx","centery","width","height"], ["Frame"]*4, ["Pixel"]*4, clippath + "/analysis/post/" + modelinfo[j] + "/" + param[k,0] + "/overtimecen")

                #aspect ratio representation
                #plot and save yolo bboxes
                h.timeplot(h.corasp(results), info[i][2] + "-results", [[0,1920],[0,1080],[0,1080],[0,2]], ["centerx","centery","width","aspect ratio"], ["Frame"]*4, ["Pixel"]*4, clippath + "/analysis/post/" + modelinfo[j] + "/" + param[k,0] + "/overtimeasp")


                plt.close("all")


    #plot errors after algorithms
    for j in range(modelinfo.shape[0]):
        for k in range(param.shape[0]):

            totalerrorcor = []
            totalerrorcen = []
            totalerrorasp = []
            #loop over clips and extract bboxes and gt --> then sum up error
            for i in range(np.shape(info)[0]):
                 
                clippath = (datapath + mainvid + "/" + info[i][2])
                #read bboxes
                resultscor = h.readmaarray(clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + param[k,0] + "/" + modelinfo[j] + param[k,0])
                gtcor = h.readmaarray(clippath + "/groundtruth/gt")
                resultscen = h.corcen(resultscor)
                gtcen = h.corcen(gtcor[:,2:6])
                resultsasp = h.corasp(resultscor)
                gtasp = h.corasp(gtcor[:,2:6])

                errorcor = h.error(resultscor, gtcor[:,2:6])
                totalerrorcor = h.apperror(errorcor,totalerrorcor)
                errorcen = h.error(resultscen, gtcen)
                totalerrorcen = h.apperror(errorcen,totalerrorcen)
                errorasp = h.error(resultsasp, gtasp)
                totalerrorasp = h.apperror(errorasp,totalerrorasp)

                binsize = 10


                h.hist(np.array(totalerrorcor),"cor", binsize, "error", ["xmin","ymin","xmax","ymax"], ["Pixel"]*4, ["norm. Anz. an Fehlern"]*4, macropath + "post/" + modelinfo[j] + "/" + param[k,0] + "/errorscor")
                h.hist(np.array(totalerrorcen),"cen", binsize, "error", ["centerx","centery","width","height"], ["Pixel"]*4, ["norm. Anz. an Fehlern"]*4, macropath + "post/" + modelinfo[j] + "/" + param[k,0] + "/errorscen")
                h.hist(np.array(totalerrorasp),"asp", binsize, "error", ["centerx","centery","width","aspect ratio"], ["Pixel"]*4, ["norm. Anz. an Fehlern"]*4, macropath + "post/" + modelinfo[j] + "/" + param[k,0] + "/errorsasp")

                #save mean and std as txt 
                meancor, stdcor = h.stats(totalerrorcor)
                meancen, stdcen = h.stats(totalerrorcen)
                meanasp, stdasp = h.stats(totalerrorasp)
                np.savetxt(macropath + "post/" + modelinfo[j] + "/" + param[k,0] + "/meanstdcor.txt", (meancor,stdcor),fmt = "%1.2f",header ="mean(first row) and std(second row) for (xmin,ymin,xmax,ymax)")
                np.savetxt(macropath + "post/" + modelinfo[j] + "/" + param[k,0] + "/meanstdcen.txt", (meancen,stdcen),fmt = "%1.2f",header ="mean(first row) and std(second row) for (centerx,centery,width,height)")
                np.savetxt(macropath + "post/" + modelinfo[j] + "/" + param[k,0] + "/meanstdasp.txt", (meanasp,stdasp),fmt = "%1.2f",header ="mean(first row) and std(second row) for (centerx,centery,width,aspect ratio)")

    print("post-analysis done")

    #loop over clips and save iou of yolo/gt as txt
    for i in range(np.shape(info)[0]):
        for j in range(modelinfo.shape[0]):
            for k in range(param.shape[0]):

                clippath = (datapath + mainvid + "/" + info[i][2])
                results = h.readmaarray(clippath + "/algorithms/kalman/" + modelinfo[j] + "/" + param[k,0] + "/" + modelinfo[j] + param[k,0])
                gt = h.readmaarray(clippath + "/groundtruth/gt")


                iou = h.iouclip(results,gt[:,2:6])
                np.savetxt(clippath + "/analysis/post/" + modelinfo[j] + "/" + param[k,0] + "/" "avgiou.txt", [iou],fmt = "%1.2f",header ="avg. iou of whole clip(only when detection exists, no \"punishment\" for no detection)")



main()
    
