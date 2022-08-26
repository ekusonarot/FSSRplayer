import cv2
import torch
from torch._C import device
import torch.backends.cudnn as cudnn
import os
import numpy as np
from pathlib import Path
import time
from models import FSRCNN
from utils import Converter, AlgorithmSelecter, calcChange, ConvertBgr2Ycbcr, Resize
import warnings
import torchvision.transforms as transforms
import inferPSNR
from ranking import Ranking, ToPatches

class Methods:

    finflag = None
    SRframes1 = None
    SRframes2 = None
    block_size =0
    scale = 0
    image_size = (0, 0)
    def __init__(self):
        self.state = None
        warnings.filterwarnings("ignore", category=UserWarning)
        cudnn.benchmark = True
        self.device = 'cpu'
        self.usethreads = torch.get_num_threads()-2 #useable thread num
        self.model = FSRCNN(scale_factor=4).to(self.device)
        state_dict = self.model.state_dict()
        for n, p in torch.load("./weights/fsrcnn_x4.pth", map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        self.model.eval()
        self.ranking = Ranking(device=self.device)
        self.ranking.load_state_dict(torch.load("weights/ranking.pth", map_location=self.device))
        self.ranking.eval()
        for i in self.ranking.parameters():
            i.required_grad = False
    

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
        fin = False
        for i in range(filenum // ign):
            if fin or faststart and limit != None:
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
                #preds = Converter.convert_bgr_to_ycbcr(fr[index])[..., 0]
                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
                SRframes[index] = np.clip(Converter.convert_ycbcr_to_bgr(output), 0.0, 255.0).astype(np.uint8)
                isSR[index] = True
                timecount = time.perf_counter() - start_time 
                if Methods.finflag == 1 or limit != None and timecount > limit:
                    SRnum += i*ign + j + 1
                    fin = True
                    break

        if fin == False and Methods.finflag == 0:
            SRnum = filenum

        Changenum, self.state = calcChange(isSR, self.state)
        return filenum, SRnum, Changenum

    #####  myFSSR  #####
    def FSSRv2(self, frames, SRframes, algonum = 2, ign = 10, fps = 60., limit = None, faststart = False, F=False):
        padding = 7
        m = 0
        block_size = Methods.block_size
        scale = Methods.scale
        h_index = Methods.image_size[0]//block_size
        w_index = Methods.image_size[1]//block_size

        torch.set_num_threads(self.usethreads) #Determine num of threads  
        filenum = len(SRframes)*h_index*w_index

        algorithm = ['ORB','AGAST','FAST','MSER','AKAZE','BRISK','KAZE','BLOB']
        isSR = [False] * filenum
        start_time = time.perf_counter()

        ##########  Ranking ##########
        start = time.perf_counter()
        finder = AlgorithmSelecter(algonum)

        keysum = 0
        keylist = []
        #for index in range(filenum):
        #    x = index//(h_index*w_index)                                                  
        #    y = index%(h_index*w_index)//w_index*block_size
        #    z = index%(h_index*w_index)%w_index*block_size
        #    kp = finder.detect(frames[x][y:y+block_size,z:z+block_size,:])
        #    keylist = keylist + [len(kp)]
        ####### ↑↓ 動作は同じだが下のほうが早い
        
        keylist = [len(finder.detect(frames[index//(h_index*w_index)][index%(h_index*w_index)//w_index*block_size:index%(h_index*w_index)//w_index*block_size+block_size,index%(h_index*w_index)%w_index*block_size:index%(h_index*w_index)%w_index*block_size+block_size,:]))
            for index in range(filenum)
        ]
        np_keylist = np.array(keylist)
        ranklist = np.argsort(-np_keylist)
        if F == False:
            ranklist = np.arange(0, filenum)
        print(time.perf_counter() - start)
        ########## SuperResolution ##########

        bic_time = time.perf_counter() 
        for i, image in enumerate(frames, 0):
            SRframes[i] = cv2.resize(image, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)

        SRnum = 0
        fin = False
        if padding > 0:
            frames = np.pad(np.array(frames), padding, "constant")[padding:-padding,:,:,padding:-padding]
        
        for i in range(filenum // ign):
            if fin or faststart and limit != None:
                break
            indexlist = ranklist[i*ign:i*ign+ign]
            Luminance = {}
            if padding >= 0:
                Luminance = np.array([Converter.convert_bgr_to_y(frames[index//(h_index*w_index)]
                    [index%(h_index*w_index)//w_index*block_size : index%(h_index*w_index)//w_index*block_size+block_size+padding*2,
                    index%(h_index*w_index)%w_index*block_size : index%(h_index*w_index)%w_index*block_size+block_size+padding*2,:].
                        astype(np.float32)) for index in indexlist])
            else:
                Luminance = np.array([Converter.convert_bgr_to_y(frames[index//(h_index*w_index)]
                    [index%(h_index*w_index)//w_index*block_size-padding : index%(h_index*w_index)//w_index*block_size+block_size+padding,
                    index%(h_index*w_index)%w_index*block_size-padding : index%(h_index*w_index)%w_index*block_size+block_size+padding,:].
                        astype(np.float32)) for index in indexlist])

            ycbcr = np.array([Converter.convert_bgr_to_ycbcr(SRframes[index//(h_index*w_index)]
                [index%(h_index*w_index)//w_index*block_size*scale:index%(h_index*w_index)//w_index*block_size*scale+block_size*scale,
                index%(h_index*w_index)%w_index*block_size*scale:index%(h_index*w_index)%w_index*block_size*scale+block_size*scale,:].astype(np.float32)) for index in indexlist])
            Luminance = torch.from_numpy(Luminance).to(self.device)
            Luminance = Luminance.unsqueeze(1)
            
            with torch.no_grad():
                preds = self.model(Luminance).mul(255.0)
            preds = preds.cpu().numpy().squeeze(1)

            if padding > 0:
                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 3, 0])
            elif padding < 0:
                output = np.array([preds, ycbcr[:,-padding*scale:padding*scale, -padding*scale:padding*scale, 1], ycbcr[:,-padding*scale:padding*scale, -padding*scale:padding*scale, 2]]).transpose([1, 2, 3, 0])
            else:
                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 3, 0])
            
            for j, index in enumerate(indexlist):
                x = index//(h_index*w_index)                                                  
                y = index%(h_index*w_index)//w_index*block_size*scale
                z = index%(h_index*w_index)%w_index*block_size*scale
                if padding < 0:
                    if m == 0:
                        SRframes[x][y-padding*scale:y+block_size*scale+padding*scale,z-padding*scale:z+block_size*scale+padding*scale,:] =\
                             np.clip(Converter.convert_ycbcr_to_bgr(output[j]), 0.0, 255.0).astype(np.uint8)
                    else:
                        SRframes[x][y-padding*scale+m:y+block_size*scale+padding*scale-m,z-padding*scale+m:z+block_size*scale+padding*scale-m,:] =\
                             np.clip(Converter.convert_ycbcr_to_bgr(output[j,m:-m,m:-m,:]), 0.0, 255.0).astype(np.uint8)
                else:
                    if m == 0:
                        SRframes[x][y:y+block_size*scale,z:z+block_size*scale,:] =\
                             np.clip(Converter.convert_ycbcr_to_bgr(output[j]), 0.0, 255.0).astype(np.uint8)
                    else:
                        SRframes[x][y+m:y+block_size*scale-m,z+m:z+block_size*scale-m,:] =\
                             np.clip(Converter.convert_ycbcr_to_bgr(output[j,m:-m,m:-m,:]), 0.0, 255.0).astype(np.uint8)
                isSR[index] = True
                timecount = time.perf_counter() - start_time
                if Methods.finflag == 1 or limit != None and timecount > limit:
                    SRnum += i*ign + j + 1
                    fin = True
                    break
            #'''

        if fin == False and Methods.finflag == 0:
            SRnum = filenum
        Changenum, self.state = calcChange(isSR, self.state)
        return filenum, SRnum, Changenum

    ##########  Acceleration  #############
    def AFSSR(self, frames, SRframes, algonum = 2, ign = 10, fps = 60., limit = None, faststart = False, F = False):
        padding = 7
        m = 0
        block_size = Methods.block_size
        scale = Methods.scale
        h_index = Methods.image_size[0]//block_size
        w_index = Methods.image_size[1]//block_size

        torch.set_num_threads(self.usethreads) #Determine num of threads  
        filenum = len(SRframes)*h_index*w_index

        algorithm = ['ORB','AGAST','FAST','MSER','AKAZE','BRISK','KAZE','BLOB']
        isSR = [False] * filenum
        start_time = time.perf_counter()

        ##########  Ranking ##########
        t = np.array([frames[index//(h_index*w_index)][index%(h_index*w_index)//w_index*block_size:index%(h_index*w_index)//w_index*block_size+block_size,index%(h_index*w_index)%w_index*block_size:index%(h_index*w_index)%w_index*block_size+block_size,:]
            for index in range(filenum)
        ])
        tensor = torch.FloatTensor(t).transpose(2,3).transpose(1,2)
        rank = self.ranking(tensor)
        rank = torch.argsort(rank, dim=0, descending=False)
        ranklist = rank.to('cpu').detach()
        if F:
            ranklist = np.arange(0, filenum)
        ########## SuperResolution ##########

        bic_time = time.perf_counter() 
        for i, image in enumerate(frames, 0):
            SRframes[i] = cv2.resize(image, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)

        SRnum = 0
        fin = False
        if padding > 0:
            frames = np.pad(np.array(frames), padding, "constant")[padding:-padding,:,:,padding:-padding]
        
        for i in range(filenum // ign):
            if fin or faststart and limit != None:
                break
            indexlist = ranklist[i*ign:i*ign+ign]
            Luminance = {}
            if padding >= 0:
                Luminance = np.array([Converter.convert_bgr_to_y(frames[index//(h_index*w_index)]
                    [index%(h_index*w_index)//w_index*block_size : index%(h_index*w_index)//w_index*block_size+block_size+padding*2,
                    index%(h_index*w_index)%w_index*block_size : index%(h_index*w_index)%w_index*block_size+block_size+padding*2,:].
                        astype(np.float32)) for index in indexlist])
            else:
                Luminance = np.array([Converter.convert_bgr_to_y(frames[index//(h_index*w_index)]
                    [index%(h_index*w_index)//w_index*block_size-padding : index%(h_index*w_index)//w_index*block_size+block_size+padding,
                    index%(h_index*w_index)%w_index*block_size-padding : index%(h_index*w_index)%w_index*block_size+block_size+padding,:].
                        astype(np.float32)) for index in indexlist])

            ycbcr = np.array([Converter.convert_bgr_to_ycbcr(SRframes[index//(h_index*w_index)]
                [index%(h_index*w_index)//w_index*block_size*scale:index%(h_index*w_index)//w_index*block_size*scale+block_size*scale,
                index%(h_index*w_index)%w_index*block_size*scale:index%(h_index*w_index)%w_index*block_size*scale+block_size*scale,:].astype(np.float32)) for index in indexlist])
            Luminance = torch.from_numpy(Luminance).to(self.device)
            Luminance = Luminance.unsqueeze(1)
            
            with torch.no_grad():
                preds = self.model(Luminance).mul(140.0)
            preds = preds.cpu().numpy().squeeze(1)

            if padding > 0:
                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 3, 0])
            elif padding < 0:
                output = np.array([preds, ycbcr[:,-padding*scale:padding*scale, -padding*scale:padding*scale, 1], ycbcr[:,-padding*scale:padding*scale, -padding*scale:padding*scale, 2]]).transpose([1, 2, 3, 0])
            else:
                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 3, 0])
            
            for j, index in enumerate(indexlist):
                x = index//(h_index*w_index)                                                  
                y = index%(h_index*w_index)//w_index*block_size*scale
                z = index%(h_index*w_index)%w_index*block_size*scale
                if padding < 0:
                    if m == 0:
                        SRframes[x][y-padding*scale:y+block_size*scale+padding*scale,z-padding*scale:z+block_size*scale+padding*scale,:] =\
                             np.clip(Converter.convert_ycbcr_to_bgr(output[j]), 0.0, 255.0).astype(np.uint8)
                    else:
                        SRframes[x][y-padding*scale+m:y+block_size*scale+padding*scale-m,z-padding*scale+m:z+block_size*scale+padding*scale-m,:] =\
                             np.clip(Converter.convert_ycbcr_to_bgr(output[j,m:-m,m:-m,:]), 0.0, 255.0).astype(np.uint8)
                else:
                    if m == 0:
                        SRframes[x][y:y+block_size*scale,z:z+block_size*scale,:] =\
                             np.clip(Converter.convert_ycbcr_to_bgr(output[j]), 0.0, 255.0).astype(np.uint8)
                    else:
                        SRframes[x][y+m:y+block_size*scale-m,z+m:z+block_size*scale-m,:] =\
                             np.clip(Converter.convert_ycbcr_to_bgr(output[j,m:-m,m:-m,:]), 0.0, 255.0).astype(np.uint8)
                isSR[index] = True
                timecount = time.perf_counter() - start_time
                if Methods.finflag == 1 or limit != None and timecount > limit:
                    SRnum += i*ign + j + 1
                    fin = True
                    break
            #'''

        if fin == False and Methods.finflag == 0:
            SRnum = filenum
        Changenum, self.state = calcChange(isSR, self.state)
        return filenum, SRnum, Changenum

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

    def LINEAR(self, frames, SRframes, limit = None, method = cv2.INTER_CUBIC): # You can use methods 0 = [cv2.INTER_NEAREST], 1 = [cv2.INTER_LINEAR], 2 = [cv2.INTER_CUBIC], 3 = [cv2.INTER_AREA], 4 = [cv2.INTER_LANCZOS4]    
        for i, image in enumerate(frames, 0):
            SRframes[i] = cv2.resize(image, None, fx = 4, fy = 4, interpolation = method) # shape is [BGR]
        if type(limit) != type(None):
            time.sleep(limit)
            return
        while Methods.finflag == 0:
            time.sleep(1)
        return