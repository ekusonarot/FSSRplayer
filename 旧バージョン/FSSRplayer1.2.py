import pafy 
import cv2
import os
import threading
import time

def ShowFrames(SRframes, lock):
    global finflag

    lock.acquire()
    for i in range(len(SRframes)):
        cv2.imshow(title, SRframes[i])
        out.write(SRframes[i])
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            break
    finflag = 1
    lock.release()

def Play():
    event.wait()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    while True:
        if len(SRframes1) == 0:
            break

        ShowFrames(SRframes1, lock1)

        if len(SRframes2) == 0:
            break

        ShowFrames(SRframes2, lock2)

def WaitPlaying(lock):
    global finflag

    event.set()
    event.clear()
    lock.acquire()
    finflag = 0
    lock.release()

def SRprocess(frames, SRframes):
    for i in range(len(frames)):
        SRframes[i] = cv2.resize(frames[i], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        if finflag == 1:
            break
    if len(frames) != len(SRframes):
        del SRframes[len(frames):]
        print("len(SRframes) = {}".format(len(SRframes)))

def SR():
    print("SR frames1 start")
    SRprocess(frames1, SRframes1)
    event.set()
    while True:
        print("SR frames2 start")
        SRprocess(frames2, SRframes2)
        if len(frames2) == 0:
            break
        WaitPlaying(lock1)

        print("SR frames1 start")
        SRprocess(frames1, SRframes1)
        if len(frames1) == 0:
            break
        WaitPlaying(lock2)

def readframes(frames, avoidDrop = False):
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
            if avoidDrop == True:
                print("Can't access frames of video")
                exit()
            elif i == 0 and frames == frames1:
                frames[i] = frames2[bufframenum-1]
            elif i == 0 and frames == frames2:
                frames[i] = frames1[bufframenum-1]
            else:
                frames[i] = frames[i-1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1.0 )


if __name__ == '__main__':
    event = threading.Event()
    lock1 = threading.Lock()
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

    print("bufframenum = {}".format(bufframenum))

    frames1 = [None] * bufframenum
    frames2 = [None] * bufframenum
    SRframes1 = [None] * bufframenum
    SRframes2 = [None] * bufframenum

    title = vPafy.title
    savepath = "{}.mp4".format(title)

    out = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    wait = int(1000./fps) # msec (waiting time between playing frame)

    readframes(frames1, avoidDrop = True)
    print("Read frames1")

    print("Start multithread")
    SRthread = threading.Thread(target=SR)
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
    cv2.destroyAllWindows()