import cv2
import torch
from torch._C import device
import torch.backends.cudnn as cudnn
import os
import numpy as np
from pathlib import Path
import time
from models import FSRCNN
from utils import Converter, AlgorithmSelecter, calcChange
import warnings
import classification
import torchvision.transforms as transforms

class Methods:

    finflag = None
    SRframes1 = None
    SRframes2 = None
    def __init__(self):
        self.state = None
        warnings.filterwarnings("ignore", category=UserWarning)
        cudnn.benchmark = True
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.model = FSRCNN(scale_factor=4).to(self.device)
        state_dict = self.model.state_dict()

        for n, p in torch.load("./weights/fsrcnn_x4.pth", map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        self.usethreads = torch.get_num_threads()-2 #useable thread num
        self.model.eval()
        ###### modified #####
        complex_model = FSRCNN(scale_factor=4, num_channels=3).to(self.device)
        complex_model.load_state_dict(torch.load("./weights/x4_3ch_fsrcnn_9.pth", map_location=torch.device(self.device)))
        light_model = FSRCNN(scale_factor=4, num_channels=3, d=32, m=1).to(self.device)
        light_model.load_state_dict(torch.load("./weights/x4_3ch_lightfsrcnn_9.pth", map_location=torch.device(self.device)))
        upsample = torch.nn.Upsample(scale_factor=4)
        self.accelerated_model = classification.CategoricalCNN(light_model, complex_model, input_shape=(3,140,260), device=self.device).to(self.device)
        self.accelerated_model.load_state_dict(torch.load("./weights/test.pth", map_location=torch.device(self.device)), strict=False)
        self.accelerated_model.eval()
        #####################

    def FSSR(self, frames, SRframes, algonum = 2, ign = 10, fps = 60., limit = None, faststart = False):
        torch.set_num_threads(self.usethreads) #Determine num of threads   
        filenum = len(frames)

        algorithm = ['ORB','AGAST','FAST','MSER','AKAZE','BRISK','KAZE','BLOB']
        isSR = [False] * filenum
        start_time = time.perf_counter()

        ##########  Ranking ##########
        finder = AlgorithmSelecter(algonum)

        keysum = 0
        keylist = []

        for i, image in enumerate(frames, 1):

            kp = finder.detect(image)
            keysum += len(kp)

            if i % ign == 0:
                keylist = keylist + [keysum]
                keysum = 0

            if i == len(frames) and i % ign != 0:
                keysum = int(keysum * (ign / (i % ign)))
                keylist = keylist + [keysum]

        np_keylist = np.array(keylist)
        ranklist = np.argsort(-np_keylist)

        ########## SuperResolution ##########

        bic_time = time.perf_counter() 
        for i, image in enumerate(frames, 0):
            SRframes[i] = cv2.resize(image, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)

        SRnum = 0
        for i in range(filenum // ign):
            if faststart and limit != None:
                break
            for j in range(0, ign):
                index = ranklist[i] * ign + j
                if index >= filenum:
                    SRnum = -ign + j
                    continue
                image = frames[index].astype(np.float32)
                bicubic = SRframes[index].astype(np.float32)
                Luminance = Converter.convert_bgr_to_y(image)
                Luminance = torch.from_numpy(Luminance).to(self.device)
                Luminance = Luminance.unsqueeze(0).unsqueeze(0)

                ycbcr = Converter.convert_bgr_to_ycbcr(bicubic)

                with torch.no_grad():
                    preds = self.model(Luminance).mul(255.0)

                preds = preds.cpu().numpy().squeeze(0).squeeze(0)
                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
                SRframes[index] = np.clip(Converter.convert_ycbcr_to_bgr(output), 0.0, 255.0).astype(np.uint8)
                isSR[index] = True
                timecount = time.perf_counter() - start_time 
                if Methods.finflag == 1 or limit != None and timecount > limit:
                    SRnum += i*ign + j + 1
                    break
            if Methods.finflag == 1 or limit != None and timecount > limit:
                break

        if Methods.finflag == 0:
            SRnum = filenum
        
        Changenum, self.state = calcChange(isSR, self.state)
        return SRnum, Changenum

    ##########  Acceleration  #############
    def AFSSR(self, frames, SRframes, algonum = 2, ign = 10, fps = 60., limit = None, faststart = False):
        torch.set_num_threads(self.usethreads) #Determine num of threads   
        filenum = len(frames)

        algorithm = ['ORB','AGAST','FAST','MSER','AKAZE','BRISK','KAZE','BLOB']
        isSR = [False] * filenum
        start_time = time.perf_counter()

        ##########  Ranking ##########
        finder = AlgorithmSelecter(algonum)

        keysum = 0
        keylist = []

        for i, image in enumerate(frames, 1):

            kp = finder.detect(image)
            keysum += len(kp)

            if i % ign == 0:
                keylist = keylist + [keysum]
                keysum = 0

            if i == len(frames) and i % ign != 0:
                keysum = int(keysum * (ign / (i % ign)))
                keylist = keylist + [keysum]

        np_keylist = np.array(keylist)
        ranklist = np.argsort(-np_keylist)

        ########## SuperResolution ##########

        bic_time = time.perf_counter() 
        for i, image in enumerate(frames, 0):
            SRframes[i] = cv2.resize(image, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)

        SRnum = 0
        for i in range(filenum // ign):
            if faststart and limit != None:
                break
            for j in range(0, ign):
                index = ranklist[i] * ign + j
                if index >= filenum:
                    SRnum = -ign + j
                    continue
                image = frames[index].astype(np.float32)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)/255
                bicubic = SRframes[index].astype(np.float32)
                image = torch.from_numpy(image).to(self.device)
                image = transforms.Resize((140, 260))(image)
                image = image.unsqueeze(0)
                with torch.no_grad():
                    preds, _ = self.accelerated_model(image)
                    preds = preds*255

                preds = preds.cpu().numpy().squeeze(0).transpose(1, 2, 0)
                preds = cv2.cvtColor(preds, cv2.COLOR_RGB2BGR)
                SRframes[index] = preds.astype(np.uint8)
                isSR[index] = True
                timecount = time.perf_counter() - start_time 
                if Methods.finflag == 1 or limit != None and timecount > limit:
                    SRnum += i*ign + j + 1
                    break
            if Methods.finflag == 1 or limit != None and timecount > limit:
                break

        if Methods.finflag == 0:
            SRnum = filenum
        
        Changenum, self.state = calcChange(isSR, self.state)
        return SRnum, Changenum

    def NSSR(self, frames, SRframes, fps = 60., limit = None, faststart = False):
        torch.set_num_threads(self.usethreads) #Determine num of threads
        filenum = len(frames)

        start_time = time.perf_counter()

        ########## Super Resolution ##########

        for i, image in enumerate(frames, 0):
            SRframes[i] = cv2.resize(image, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)

        timecount = time.perf_counter() - start_time #bicubic 

        SRnum = 0
        for i in range(filenum):
            if faststart and limit != None:
                break
            image = frames[i].astype(np.float32)
            bicubic = SRframes[i].astype(np.float32)
            Luminance = Converter.convert_bgr_to_y(image)
            Luminance = torch.from_numpy(Luminance).to(self.device)
            Luminance = Luminance.unsqueeze(0).unsqueeze(0)

            ycbcr = Converter.convert_bgr_to_ycbcr(bicubic)

            with torch.no_grad():
                preds = self.model(Luminance).mul(255.0)

            preds = preds.cpu().numpy().squeeze(0).squeeze(0)
            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            SRframes[i] = np.clip(Converter.convert_ycbcr_to_bgr(output), 0.0, 255.0).astype(np.uint8)
            timecount = time.perf_counter() - start_time
            if Methods.finflag == 1 or limit != None and timecount > limit:
                SRnum = i + 1
                break
        if Methods.finflag == 0:
            SRnum = filenum
        return SRnum

    def LINEAR(self, frames, SRframes, method = cv2.INTER_CUBIC): # You can use methods 0 = [cv2.INTER_NEAREST], 1 = [cv2.INTER_LINEAR], 2 = [cv2.INTER_CUBIC], 3 = [cv2.INTER_AREA], 4 = [cv2.INTER_LANCZOS4]    
        for i, image in enumerate(frames, 0):
            SRframes[i] = cv2.resize(image, None, fx = 4, fy = 4, interpolation = method) # shape is [BGR]