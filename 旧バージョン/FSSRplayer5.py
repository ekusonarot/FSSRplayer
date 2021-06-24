import pafy 
import cv2
import os
import threading
import time
from methods_enc import Methods
from utils import millisec
import psutil
import time

def ShowFrames(SRframes, lock):
    lock.acquire()
    start = millisec()
    seektime = wait
    for i in range(len(SRframes)):
        #cv2.imshow(title, SRframes[i]) #cut if no play
        now = millisec()
        waittime = 1 if seektime == now - start or seektime < now - start else seektime - (now - start)
        if seektime < now - start - 200:
            print("Warning : playing speed may be late")
        time.sleep(waittime / 1000.)
        #if cv2.waitKey(waittime) & 0xFF == ord('q'): #cut if no play
        #    break #cut if no play
        out.write(SRframes[i])
        SRframes[i] = None
        seektime += wait
    Methods.finflag = 1
    lock.release()

def Play():
    event.wait()
    #cv2.namedWindow(title, cv2.WINDOW_NORMAL)#cut if no play
    while True:
        if len(Methods.SRframes1) == 0:
            print("Play thread fin.")
            #cv2.destroyWindow(title)##cut if no play
            break

        ShowFrames(Methods.SRframes1, lock1)

        if len(Methods.SRframes2) == 0:
            print("Play thread fin.")
            #cv2.destroyWindow(title)##cut if no play
            break
            
        ShowFrames(Methods.SRframes2, lock2)

def WaitPlaying(lock):

    event.set()
    lock.acquire()
    Methods.finflag = 0
    lock.release()

def SRprocess(frames, SRframes, method, ign, limit = None):
    global batchcount, SRsum, Changesum

    if len(frames) != len(SRframes):
        del SRframes[len(frames):]
        print("len(SRframes) = {}".format(len(SRframes)))

    if method == "FSSR":
        SRnum, Changenum = SRmethods.FSSR(frames, SRframes, algonum, ign, fps, limit)
        f.write("{},{},{}\n".format(batchcount, SRnum, Changenum))
        SRsum += SRnum
        Changesum += Changenum
    elif method == "NSSR":
        SRnum = SRmethods.NSSR(frames, SRframes, fps, limit)
        f.write("{},{}\n".format(batchcount, SRnum))
        SRsum += SRnum
    elif method == "LINEAR":
        SRmethods.LINEAR(frames, SRframes)
    else:
        print("[ERROR] Please select a method")
        exit()
    
    batchcount += 1


def SR(method, ign, buftime):
    global batchcount, SRsum, Changesum
    #lim = None
    batchcount = 1
    SRsum = 0
    Changesum = 0
    print("SR frames1 start")
    SRprocess(frames1, Methods.SRframes1, method, ign, limit=buftime)
    Methods.finflag = 0
    event.set()
    while True:

        #if len(frames2) != bufframenum:# for fair eval
        #    lim = len(frames2) / fps
        #    print("lim={}".format(lim))
        
        print("SR frames2 start")
        SRprocess(frames2, Methods.SRframes2, method, ign)

        WaitPlaying(lock1)
        if len(frames1) == 0:
            del Methods.SRframes1[0:]

            if method == "FSSR":
                f.write("Total,{},{}\n".format(SRsum, Changesum))
            elif method == "NSSR":
                f.write("Total,{}\n".format(SRsum))
            break

        #if len(frames1) != bufframenum:# for fair eval
        #    lim = len(frames1) / fps 

        print("SR frames1 start")
        SRprocess(frames1, Methods.SRframes1, method, ign)

        WaitPlaying(lock2)
        if len(frames2) == 0:
            del Methods.SRframes2[0:]

            if method == "FSSR":
                f.write("Total,{},{}\n".format(SRsum, Changesum))
            elif method == "NSSR":
                f.write("Total,{}\n".format(SRsum))

            break

def readframes(frames, avoidDrop = False):
    for i in range(len(frames)):
        #print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret,frames[i] = cap.read()
        if ret == False:
            print("1OUT")
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret,frames[i] = cap.read()
            if ret == False:
                print("2OUT")
                del frames[i:]
                break

#if __name__ == '__main__':
def videoplay(method, video, ign = 10, buftime = 20.):
    #method = "FSSR " # FSSR or NSSR or LINEAR
    #video = "BBB24fps/BBB24fps_144p.mp4"
    #ign = 10
    #buftime = 20.

    global SRmethods, event, lock1, lock2, title, algonum
    algonum = 2 #FAST
    SRmethods = Methods()
    event = threading.Event()
    lock1 = threading.Lock()
    lock2 = threading.Lock()

    #url = "http://192.168.1.142/testgt.mp4"
    server = "http://192.168.1.142"
    url = os.path.join(server,video)
    videoname = os.path.splitext(os.path.basename(url))[0]
    title = os.path.splitext(os.path.basename(url))[0] + " - " + method

    """ For Youtube play
    url = 'https://youtu.be/09R8_2nJtjg' # Sugar
    #url = 'https://youtu.be/YXH2j16PEiI'
    vPafy = pafy.new(url)
    play = vPafy.videostreams[2]
    print(play)
    #start the video
    cap = cv2.VideoCapture(play.url)
    """
    global cap, fps, width, height, framenum, bufframenum
    cap = cv2.VideoCapture(url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*4)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*4)
    framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bufframenum = int(buftime * fps)

    print("bufframenum = {}".format(bufframenum))

    global frames1, frames2, out, wait, f
    frames1 = [None] * bufframenum
    frames2 = [None] * bufframenum
    Methods.SRframes1 = [None] * bufframenum
    Methods.SRframes2 = [None] * bufframenum
    Methods.finflag = 2
    print("len(SRframes1) = {}".format(len(Methods.SRframes1)))
    print("len(SRframes2) = {}".format(len(Methods.SRframes2)))

    dirname = os.path.dirname(video)
    savedir = os.path.join("./videolog", dirname)
    os.makedirs(savedir, exist_ok="True")
    savevideo = "{}{}{}_{}.mp4".format(savedir, os.sep, videoname, method)
    evalfile = "{}{}{}_{}_SRcount.csv".format(savedir, os.sep, videoname, method)
    #if ign != 10:
    savevideo = "{}{}{}_{}_{}ign.mp4".format(savedir, os.sep, videoname, method, ign)
    evalfile = "{}{}{}_{}_{}ign_SRcount.csv".format(savedir, os.sep, videoname, method, ign)
    #if buftime != 20.:
    #savevideo = "{}{}{}_{}_{}segtime.mp4".format(savedir, os.sep, videoname, method, int(buftime))
    #evalfile = "{}{}{}_{}_{}segtime_SRcount.csv".format(savedir, os.sep, videoname, method, int(buftime))


    if method == "FSSR":
        f = open(evalfile, mode='w')
        f.write("batch, SRnum, Changenum\n")
    elif method == "NSSR":
        f = open(evalfile, mode='w')
        f.write("batch, SRnum\n")
    out = cv2.VideoWriter(savevideo, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    wait = int(1000./fps) # msec (waiting time between playing frame)

    print("wait = {}".format(wait))
    readframes(frames1, avoidDrop = True)
    print("Read frames1")

    print("Start multithread")
    SRthread = threading.Thread(target=SR, args=([method, ign, buftime]))
    PLAYthread = threading.Thread(target=Play)
    SRthread.start()
    PLAYthread.start()
    while (True):

        readframes(frames2)
        print("Read frames2")

        event.wait()
        event.clear()

        readframes(frames1)
        print("Read frames1")

        if len(frames1) == 0 or len(frames2) == 0:
            break

        event.wait()
        event.clear()

    print("Waiting SR thread...")
    SRthread.join()
    print("Waiting PLAY thread...")
    PLAYthread.join()
    cap.release()
    out.release()
    if method == "FSSR" or method == "NSSR":
        f.close()
    return