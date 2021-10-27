import cv2
import os
from utils import LPIPSpreprocess
import lpips
from skimage.metrics import structural_similarity as ssim
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
    HRvideo = cv2.VideoCapture(HRvideopath)
    if HRvideo.isOpened() == False:
        print("GT path is incorrect.")
        exit()
    print("\n----- Evaluation [{}] Start -----\n".format(os.path.basename(SRvideopath)))
    fps = HRvideo.get(cv2.CAP_PROP_FPS)

    #lpips_model = lpips.LPIPS(net='alex')
    lpips_model = lpips.LPIPS(net='alex').cuda() # Use CUDA gpu

    sumPSNR = 0.
    sumSSIM = 0.
    sumLPIPS = 0.
    minPSNR = 100.
    minSSIM = 100.
    minLPIPS = 100.
    maxPSNR = 0.
    maxSSIM = 0.
    maxLPIPS = 0.

    for i in range(framenum):
        bic_time = time.perf_counter() 
        retSR, SRimage = SRvideo.read()
        retHR, HRimage = HRvideo.read()   
        if retSR == False or retHR == False:
            print("\n ERROR Happened ")
            break

        PSNRvalue = cv2.PSNR(SRimage, HRimage)
        SSIMvalue = ssim(SRimage, HRimage, multichannel = True)
        #LPIPSvalue = lpips_model(LPIPSpreprocess(SRimage), LPIPSpreprocess(HRimage))[0,0,0,0].item()
        LPIPSvalue = lpips_model(LPIPSpreprocess(SRimage).cuda(), LPIPSpreprocess(HRimage).cuda()) # Use CUDA gpu

        sumPSNR += PSNRvalue
        sumSSIM += SSIMvalue
        sumLPIPS += float(LPIPSvalue)
        if PSNRvalue > maxPSNR:
            maxPSNR = PSNRvalue
        if PSNRvalue < minPSNR:
            minPSNR = PSNRvalue
        if SSIMvalue > maxSSIM:
            maxSSIM = SSIMvalue
        if SSIMvalue < minSSIM:
            minSSIM = SSIMvalue
        if float(LPIPSvalue) > maxLPIPS:
            maxLPIPS = float(LPIPSvalue)
        if float(LPIPSvalue) < minLPIPS:
            minLPIPS = float(LPIPSvalue)
        del LPIPSvalue
        e_time = time.perf_counter() - bic_time
        e_time = e_time * (framenum-i)
        H = int(e_time//3600)
        M = int(e_time%3600//60)
        S = int(e_time%3600%60)
        print("\r[{}/{}] frames evaluation done ETA:{:02}:{:02}:{:02}s.".format(i+1, framenum, H, M, S), end="")
    avgPSNR = sumPSNR/framenum
    avgSSIM = sumSSIM/framenum
    avgLPIPS = sumLPIPS/framenum

    print("\n ---Similalities between [{}] and [{}] ---\n   avgPSNR  : {}\n   avgSSIM  : {}\n   avgLPIPS : {}"\
.format(os.path.basename(SRvideopath), os.path.basename(HRvideopath),avgPSNR,avgSSIM,avgLPIPS))
    print("\n ------------------------------------------\n\
    \t\tmax\tmin\nPSNR:\t{}\t{}\nSSIM:\t{}\t{}\nLPIPS:\t{}\t{}".format(maxPSNR,minPSNR,maxSSIM,minSSIM,maxLPIPS,minLPIPS))
