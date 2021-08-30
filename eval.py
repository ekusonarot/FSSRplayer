import cv2
import os
from utils import LPIPSpreprocess
import lpips
from skimage.metrics import structural_similarity as ssim
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="set Input video path", type=str, required=True)
    parser.add_argument("-gt","--ground_truth", help="set Ground-Truth video path", type=str, required=True)
    args = parser.parse_args()

    SRvideopath = args.input
    print(SRvideopath)
    HRvideopath = args.ground_truth
    SRvideo = cv2.VideoCapture(SRvideopath)
    if SRvideo.isOpened() == False:
        print("Input path is incorrect.")
        exit()
    HRvideo = cv2.VideoCapture(HRvideopath)
    if HRvideo.isOpened() == False:
        print("GT path is incorrect.")
        exit()
    print("\n----- Evaluation [{}] Start -----\n".format(os.path.basename(SRvideopath)))
    fps = SRvideo.get(cv2.CAP_PROP_FPS)
    framenum = int(SRvideo.get(cv2.CAP_PROP_FRAME_COUNT))

    lpips_model = lpips.LPIPS(net='alex')
    # lpips_model = lpips.LPIPS(net='alex').cuda() # Use CUDA gpu

    sumPSNR = 0.
    sumSSIM = 0.
    sumLPIPS = 0.

    for i in range(framenum):
        retSR, SRimage = SRvideo.read()
        retHR, HRimage = HRvideo.read()   
        if retSR == False or retHR == False:
            print("\n ERROR Happened ")
            break

        PSNRvalue = cv2.PSNR(SRimage, HRimage)
        SSIMvalue = ssim(SRimage, HRimage, multichannel = True)
        LPIPSvalue = lpips_model(LPIPSpreprocess(SRimage), LPIPSpreprocess(HRimage))[0,0,0,0].item()
        #LPIPSvalue = lpips_model(LPIPSpreprocess(SRimage).cuda(), LPIPSpreprocess(HRimage).cuda()) # Use CUDA gpu

        sumPSNR += PSNRvalue
        sumSSIM += SSIMvalue
        sumLPIPS += LPIPSvalue
        print("\r[{}/{}] frames evaluation done.".format(i+1, framenum), end="")
    avgPSNR = sumPSNR/framenum
    avgSSIM = sumSSIM/framenum
    avgLPIPS = sumLPIPS/framenum

    print("\n ---Similalities between [{}] and [{}] ---\n   avgPSNR  : {}\n   avgSSIM  : {}\n   avgLPIPS : {}"\
.format(os.path.basename(SRvideopath), os.path.basename(HRvideopath),avgPSNR,avgSSIM,avgLPIPS))
