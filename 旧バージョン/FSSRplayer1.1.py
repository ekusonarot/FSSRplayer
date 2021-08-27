import pafy 
import cv2
import os
#from multiprocessing import Process, Pipe
import threading
import time

def Play():
    global finflag
    event.wait()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    while True:

        print(len(SRframes))
        if len(SRframes) == 0:
            break

        lock.acquire()
        for i in range(len(SRframes)):
            cv2.imshow(title, SRframes[i])
            out.write(SRframes[i])
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        finflag = 1
        lock.release()

        print(len(SRframes2))
        if len(SRframes2) == 0:
            break

        lock2.acquire()
        for i in range(len(SRframes2)):
            cv2.imshow(title, SRframes2[i])
            out.write(SRframes2[i])
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        finflag = 1
        lock2.release()

def SR():
    global finflag
    global SRframes
    global SRframes2
    print("SR frames start")
    for i in range(len(frames)):

        SRframes[i] = cv2.resize(frames[i], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)

    event.set()
    while True:

        print("SR frames2 start")
        for i in range(len(frames2)):

            SRframes2[i] = cv2.resize(frames2[i], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
    
            if finflag == 1:
                break

        if len(frames2) != len(SRframes2):
            del SRframes2[len(frames2):]
            print("len(SRframes2) = {}".format(len(SRframes2)))
            if len(frames2) == 0:
                break

        event.set()
        event.clear()
        lock.acquire()
        finflag = 0
        lock.release()


        print("SR frames start")
        for i in range(len(frames)):

            SRframes[i] = cv2.resize(frames[i], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)

            if finflag == 1:
                break

        if len(frames) != len(SRframes):
            del SRframes[len(frames):]
            print("len(SRframes) = {}".format(len(SRframes)))
            if len(frames) == 0:
                break

        event.set()
        event.clear()     
        lock2.acquire()
        finflag = 0
        lock2.release()


if __name__ == '__main__':

    event = threading.Event()
    lock = threading.Lock()
    lock2 = threading.Lock()
    finflag = 0
    buftime = 20.
    #url = 'https://youtu.be/09R8_2nJtjg' # Sugar
    url = 'https://youtu.be/YXH2j16PEiI'
    vPafy = pafy.new(url)
    play = vPafy.videostreams[0]
    print(play)

    #start the video
    cap = cv2.VideoCapture(play.url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*4)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*4)
    framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bufframenum = int(buftime * fps)

    ###############

    print("bufframenum = {}".format(bufframenum))

    frames = [None] * bufframenum
    frames2 = [None] * bufframenum
    SRframes = [None] * bufframenum
    SRframes2 = [None] * bufframenum

    ###############

    title = vPafy.title
    savepath = "{}.mp4".format(title)

    out = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    wait = int(1000./fps) # msec (waiting time between playing frame)
    print(wait)
    print("Reading first segment")

    for i in range(bufframenum):
        ret,frames[i] = cap.read()
        if ret==False:
            print("Can't get frames {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            if i == 0:
                print("Can't get first frame")
            else:
                frames[i] = frames[i-1]

            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1.0 )
            break

    print("Read frames")
    print("Start multithread")
    SRthread = threading.Thread(target=SR)
    PLAYthread = threading.Thread(target=Play)
    #SRthread.setDaemon(True)
    #PLAYthread.setDaemon(True)
    SRthread.start()
    PLAYthread.start()
    Fin = False

    while (True):
        for i in range(bufframenum):
            ret,frames2[i] = cap.read()
            if ret == False:
                current = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1.0 )
                if current == cap.get(cv2.CAP_PROP_POS_FRAMES):
                    del frames2[i:]
                    print("len(frames2) = {}".format(len(frames2)))
                    break
                print("Can't get frames {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                if i == 0:
                    frames2[i] = frames[bufframenum-1]
                else:
                    frames2[i] = frames2[i-1]
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1.0 )

        print("Read frames2")

        event.wait()
        event.clear()

        for i in range(bufframenum):
            ret,frames[i] = cap.read()
            if ret == False:
                current = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1.0 )
                if current == cap.get(cv2.CAP_PROP_POS_FRAMES):
                    del frames[i:]
                    print("len(frames) = {}".format(len(frames)))
                    break
                print("Can't get frames {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                if i == 0:
                    frames[i] = frames2[bufframenum-1]
                else:
                    frames[i] = frames[i-1]
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1.0 )
        print("Read frames")
        if len(frames) == 0 or len(frames2) == 0:
            break

        event.wait()
        event.clear()

    print("Waiting SR thread...")
    SRthread.join()
    print("Waiting PLAY thread...")
    PLAYthread.join()
    cap.release()
    out.release()
    cv2.destroyAllWindows()