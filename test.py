import cv2
import os
import argparse
import time
import numpy as np
import lpips
from skimage.metrics import structural_similarity as ssim
from utils import LPIPSpreprocess

def createImage(frame, srframe, bcframe, hrframe):
    scale=4
    h_index = 9
    w_index = 16
    block_size=16
    for index in range(h_index*w_index):                  
        y = index%(h_index*w_index)//w_index*block_size*scale
        z = index%(h_index*w_index)%w_index*block_size*scale
        sr = srframe[y:y+block_size*scale,z:z+block_size*scale,:]
        bc = bcframe[y:y+block_size*scale,z:z+block_size*scale,:]
        hr = hrframe[y:y+block_size*scale,z:z+block_size*scale,:]
        if cv2.PSNR(sr,hr) < cv2.PSNR(bc,hr):
            frame[y:y+block_size*scale,z:z+block_size*scale,:] = bc
        else:
            frame[y:y+block_size*scale,z:z+block_size*scale,:] = bc

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1","--input1", help="set Input video path", type=str, required=True)
    parser.add_argument("-i2","--input2", help="set Input video path", type=str, required=True)
    parser.add_argument("-gt","--ground_truth", help="set Ground-Truth video path", type=str, required=True)
    args = parser.parse_args()

    SRvideopath = args.input1
    BCvideopath = args.input2
    HRvideopath = args.ground_truth
    '''
    SRvideopath = "./GT.mp4"
    BCvideopath = "./BIC.mp4"
    HRvideopath = "./GTBest1.mp4"
    SRvideo = cv2.VideoCapture(SRvideopath)
    if SRvideo.isOpened() == False:
        print("Input path is incorrect.")
        exit()
    BCvideo = cv2.VideoCapture(BCvideopath)
    if BCvideo.isOpened() == False:
        print("Input path is incorrect.")
        exit()
    HRvideo = cv2.VideoCapture(HRvideopath)
    if HRvideo.isOpened() == False:
        print("GT path is incorrect.")
        exit()
    fps = SRvideo.get(cv2.CAP_PROP_FPS)
    w = int(SRvideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(SRvideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framenum = int(SRvideo.get(cv2.CAP_PROP_FRAME_COUNT))

    wait = int(1000/fps)
    frame = np.zeros((h,w,3), dtype=np.uint8)
    out = cv2.VideoWriter("save.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    sumPSNR = 0.
    sumSSIM = 0.
    sumLPIPS = 0.
    minPSNR = 100.
    minSSIM = 100.
    minLPIPS = 100.
    maxPSNR = 0.
    maxSSIM = 0.
    maxLPIPS = 0.
    lpips_model = lpips.LPIPS(net='alex').cuda()
    for i in range(framenum):
        bic_time = time.perf_counter() 
        _, srframe = SRvideo.read()
        _, bcframe = BCvideo.read()
        ret, hrframe = HRvideo.read()
        if ret:
            createImage(frame,srframe,bcframe,hrframe)
            #cv2.imshow("TEST", frame)
            out.write(frame)

            PSNRvalue = cv2.PSNR(frame, hrframe)
            SSIMvalue = ssim(frame, hrframe, multichannel = True)
            LPIPSvalue = lpips_model(LPIPSpreprocess(frame).cuda(), LPIPSpreprocess(hrframe).cuda())
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
        else:
            break
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
    #cv2.destroyWindow("TEST")
