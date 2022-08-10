import cv2
import argparse
import os

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-v","--video",type=str)
    #parser.add_argument("-o","--savedir",type=str,default="./video")
    #parser.add_argument("-w","--width",type=int)
    #parser.add_argument("-j","--height",type=int)
    #args = parser.parse_args()

    videos = ["GT/GT_BBB60fps.mp4"]#,"GT/GT_HH.mp4","GT/GT_ToS.mp4"]
    #size = (256,144)
    #size = (320,180)
    #size = (480,270)
    #size = (1024,576)
    #size = (1280,720)
    #size = (1920,1080)
    sizes = [(1024,576),(1280,720),(1920,1080)]
    for video in videos:
        for size in sizes:
            lsize = (size[0]//4,size[1]//4)
            time = 600
            #gsavevideo = "{}_{}p.mp4".format(video.split('.')[0],size[1])
            #lsavevideo = "{}_{}p.mp4".format(video.split('.')[0],lsize[1])
            gsavevideo = "GT/GT_BBB24fps_{}p.mp4".format(size[1])
            lsavevideo = "GT/GT_BBB24fps_{}p.mp4".format(lsize[1])
            cap = cv2.VideoCapture(video)
            #fps = cap.get(cv2.CAP_PROP_FPS)
            fps = 24
            flag_24fps = True
            flag_30fps = False

            gout = cv2.VideoWriter(gsavevideo, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            lout = cv2.VideoWriter(lsavevideo, cv2.VideoWriter_fourcc(*'mp4v'), fps, lsize)
            i = 0
            count = 0
            while True:
                ret, frame = cap.read()
                #print("A {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                #'''
                i = i+1
                if flag_24fps or flag_30fps:
                    if i % 2 == 1:
                        ret, test = cap.read()
                        ret, test = cap.read()
                        #print("B {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                        #cv2.imshow('Frame', test)
                        #cv2.waitKey(33)
                        i = i+1
                    if flag_24fps and i % 10 == 0:
                        ret, test = cap.read()
                        #print("C {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                        #cv2.imshow('Frame', test)
                        #cv2.waitKey(33)
                        i = i+1
                #'''
                if ret == False:
                    break
                if time != None and time*fps < count:
                    break
                count += 1
                gframe = cv2.resize(frame, dsize=size)
                lframe = cv2.resize(gframe, dsize=lsize)
                gout.write(gframe)
                lout.write(lframe)
            
            cap.release()
            gout.release()
            lout.release()