import pafy 
import cv2
import os
import threading
import time
from methods import Methods
from utils import millisec
import psutil
import time
import argparse

def ShowFrames(SRframes, lock):
    lock.acquire()
    start = millisec()
    seektime = wait
    for i in range(len(SRframes)):
        cv2.imshow(title, SRframes[i])
        now = millisec()
        waittime = 1 if seektime == now - start or seektime < now - start else seektime - (now - start)
        if seektime < now - start - 200:
            print("Warning : playing speed may be late")
        if cv2.waitKey(waittime) & 0xFF == ord('q'):
            break
        out.write(SRframes[i])
        SRframes[i] = None
        seektime += wait
    Methods.finflag = 1
    lock.release()

def Play():
    event.wait()
    while True:
        if len(Methods.SRframes1) == 0:
            cv2.destroyWindow(title)
            break

        ShowFrames(Methods.SRframes1, lock1)

        if len(Methods.SRframes2) == 0:
            cv2.destroyWindow(title)
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

    if method == "FSSR":
        SRnum, Changenum = SRmethods.FSSR(frames, SRframes, algonum, ign, fps, limit, faststart)

        if outEval:
            f.write("{},{},{}\n".format(batchcount, SRnum, Changenum))

        SRsum += SRnum
        Changesum += Changenum
    elif method == "FSSRv2":
        SRnum, Changenum = SRmethods.FSSRv2(frames, SRframes, algonum, ign, fps, limit, faststart)

        if outEval:
            f.write("{},{},{}\n".format(batchcount, SRnum, Changenum))

        SRsum += SRnum
        Changesum += Changenum
    elif method == "AFSSR":
        SRnum, Changenum = SRmethods.AFSSR(frames, SRframes, algonum, ign, fps, limit, faststart)

        if outEval:
            f.write("{},{},{}\n".format(batchcount, SRnum, Changenum))

        SRsum += SRnum
        Changesum += Changenum
    elif method == "NSSR":
        SRnum = SRmethods.NSSR(frames, SRframes, fps, limit, faststart)

        if outEval:
            f.write("{},{}\n".format(batchcount, SRnum))

        SRsum += SRnum
    elif method == "BIC":
        SRmethods.LINEAR(frames, SRframes)

    else:
        print("[ERROR] Please select a method")
        exit()
    
    batchcount += 1


def SR(method, ign, buftime):
    global batchcount, SRsum, Changesum
    batchcount = 1
    SRsum = 0
    Changesum = 0
    SRprocess(frames1, Methods.SRframes1, method, ign, limit=buftime)
    Methods.finflag = 0
    event.set()
    while True:
        
        SRprocess(frames2, Methods.SRframes2, method, ign)

        WaitPlaying(lock1)
        if len(frames1) == 0:
            del Methods.SRframes1[0:]

            if outEval:
                if method == "FSSR":
                    f.write("Total,{},{}\n".format(SRsum, Changesum))
                if method == "FSSRv2":
                    f.write("Total,{},{}\n".format(SRsum, Changesum))
                if method == "AFSSR":
                    f.write("Total,{},{}\n".format(SRsum, Changesum))
                elif method == "NSSR":
                    f.write("Total,{}\n".format(SRsum))
            break

        SRprocess(frames1, Methods.SRframes1, method, ign)

        WaitPlaying(lock2)
        if len(frames2) == 0:
            del Methods.SRframes2[0:]

            if outEval:
                if method == "FSSR":
                    f.write("Total,{},{}\n".format(SRsum, Changesum))
                if method == "FSSRv2":
                    f.write("Total,{},{}\n".format(SRsum, Changesum))
                if method == "AFSSR":
                    f.write("Total,{},{}\n".format(SRsum, Changesum))
                elif method == "NSSR":
                    f.write("Total,{}\n".format(SRsum))

            break

def readframes(frames, avoidDrop = False):
    for i in range(len(frames)):
        ret,frames[i] = cap.read()
        if ret == False: # check recv-error or end-of-video
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frames[i] = cap.read()
            if ret == False:
                del frames[i:]
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--batchsize", type=int, default=10)
    parser.add_argument("-s","--segsize", type=float, default=20.)
    parser.add_argument("-m","--method", type=str, choices=['FSSR', 'FSSRv2', 'NSSR', 'AFSSR', 'BIC'], default='FSSR')
    parser.add_argument("-e","--eval", action="store_true")
    parser.add_argument("-f","--faststart", action="store_true")
    args = parser.parse_args()

    global outEval, faststart
    ign = args.batchsize
    buftime = args.segsize
    method = args.method
    outEval = args.eval
    faststart = args.faststart

    print("\n   --- Parameters ---   \n\
 Batchsize : {}\n\
 Segsize   : {}\n\
 SRmethod  : {}\n".format(ign, buftime, method))

    global SRmethods, event, lock1, lock2, title, algonum
    algonum = 2 # Select feature-detection algorithm. 2 -> FAST
    SRmethods = Methods()
    event = threading.Event()
    lock1 = threading.Lock()
    lock2 = threading.Lock()

    url = 'https://youtu.be/BBvod49uySQ' # Play video URL by Youtube
    #url = 'https://youtu.be/_1VZcrBMqLU'
    
    vPafy = pafy.new(url)
    play = vPafy.videostreams[1]
    title = vPafy.title

    global cap, fps, width, height, framenum, bufframenum

    cap = cv2.VideoCapture(play.url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*4)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*4)
    framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bufframenum = int(buftime * fps)
    time = int(framenum / fps)

    print("   --- Play video ---   \n\
 Title : {}\n\
 Size  : {}x{}\n\
 Time  : {} sec\n".format(title, int(width/4), int(height/4), time))
    global frames1, frames2, out, wait, f
    frames1 = [None] * bufframenum
    frames2 = [None] * bufframenum
    Methods.SRframes1 = [None] * bufframenum
    Methods.SRframes2 = [None] * bufframenum
    Methods.finflag = 2

    savedir = os.path.join("./videolog", vPafy.title)
    os.makedirs(savedir, exist_ok="True")
    savevideo = "{}{}{}_{}ign{}buftime.mp4".format(savedir, os.sep, method, ign, int(buftime))
    if method == "BIC":
        savevideo = "{}{}{}.mp4".format(savedir, os.sep, method)

    if outEval:
        evalfile = "{}{}{}_{}ign{}buftime_count.csv".format(savedir, os.sep, method, ign, int(buftime))
        if method == "FSSR":
            f = open(evalfile, mode='w')
            f.write("batch, SRnum, Changenum\n")
        if method == "FSSRv2":
            f = open(evalfile, mode='w')
            f.write("batch, SRnum, Changenum\n")
        if method == "AFSSR":
            f = open(evalfile, mode='w')
            f.write("batch, SRnum, Changenum\n")
        elif method == "NSSR":
            f = open(evalfile, mode='w')
            f.write("batch, SRnum\n")

    out = cv2.VideoWriter(savevideo, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    wait = int(1000./fps) # msec (waiting time between playing frame)
    readframes(frames1, avoidDrop = True)

    SRthread = threading.Thread(target=SR, args=([method, ign, buftime]))
    PLAYthread = threading.Thread(target=Play)
    SRthread.start()
    PLAYthread.start()
    while (True):

        readframes(frames2)

        event.wait()
        event.clear()

        readframes(frames1)

        if len(frames1) == 0 or len(frames2) == 0:
            break

        event.wait()
        event.clear()

    SRthread.join()
    PLAYthread.join()
    cap.release()
    out.release()
    if outEval and (method == "FSSR" or method == "NSSR"):
        f.close()