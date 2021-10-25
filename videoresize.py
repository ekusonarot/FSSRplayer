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

    videos = ["tears_of_steel_1080p.mp4","Herzmark_Homestead.mp4","BigBuckBunny.mp4"]
    #size = (256,144)
    #size = (320,180)
    #size = (480,270)
    #size = (1024,576)
    #size = (1280,720)
    #size = (1920,1080)
    sizes = [(1024,576),(1280,720),(1920,1080)]
    for video in videos:
        for size in sizes:
            time = 600
            savedir = "video"
            savedir = os.path.join(savedir,video)
            os.makedirs(savedir, exist_ok="True")
            savevideo = "{}/{}_{}.avi".format(savedir,video,size)

            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            out = cv2.VideoWriter(savevideo, cv2.VideoWriter_fourcc(*'I420'), fps, size)
            while True:
                ret, frame = cap.read()
                if ret == False:
                    break
                if time != None and time*fps < cap.get(cv2.CAP_PROP_POS_FRAMES):
                    break
                frame = cv2.resize(frame, dsize=size)
                out.write(frame)
            
            cap.release()
            out.release()