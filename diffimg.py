import cv2
import os
import argparse
import time

class Videolist:
    def __init__(self, videolist):
        self._list = videolist
        self._count = 0
    def read(self):
        if self._count == len(self._list):
            return False, None
        i = self._count
        self._count += 1
        return True, self._list[i]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="set Input video path", type=str, required=True)
    parser.add_argument("-gt","--ground_truth", help="set Ground-Truth video path", type=str, required=True)
    args = parser.parse_args()

    SRvideopath = args.input
    SRvideo = cv2.VideoCapture(SRvideopath)
    if SRvideo.isOpened() == False:
        print("Input path is incorrect.")
        exit()
    framenum = int(SRvideo.get(cv2.CAP_PROP_FRAME_COUNT))

    HRvideopath = args.ground_truth
    #'''
    HRvideo = cv2.VideoCapture(HRvideopath)
    if HRvideo.isOpened() == False:
        print("GT path is incorrect.")
        exit()
    fps = HRvideo.get(cv2.CAP_PROP_FPS)
    wait = int(1000/fps)
    #'''
    while True:
        ret, srframe = SRvideo.read()
        ret, hrframe = HRvideo.read()
        if ret:
            difframe = hrframe - srframe
            cv2.imshow("diff", difframe)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyWindow("diff")
