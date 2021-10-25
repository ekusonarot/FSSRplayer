import torch
import torch.optim
from torch.utils.data import Dataset
import math
import cv2
from models import FSRCNN
from utils import Converter
import numpy as np
import torchvision
import math

class dataset(Dataset):
    def __init__(self, input_path, output_path):
        input_cap = cv2.VideoCapture(input_path)
        output_cap = cv2.VideoCapture(output_path)
        self._numbers = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._i = 0
        self.SRimages = [None]*self._numbers
        self.HRimages = [None]*self._numbers
        for i in range(self._numbers):
            _, self.SRimages[i] = input_cap.read()
            _, self.HRimages[i] = output_cap.read()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = FSRCNN(scale_factor=4).to(self.device)
        state_dict = self.model.state_dict()
        for n, p in torch.load("./weights/fsrcnn_x4.pth", map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        self.model.eval()
        
        self.h_index = 9
        self.w_index = 16
        self.scale = 4
        block_size = 16
        self.inputs = [None]*self._numbers*self.h_index*self.w_index
        self.psnrs = [None]*self._numbers*self.h_index*self.w_index
        for i in range(self._numbers*self.h_index*self.w_index):
            self.inputs[i] = self.SRimages[i//(self.h_index*self.w_index)][i%(self.h_index*self.w_index)//self.w_index*block_size:i%(self.h_index*self.w_index)//self.w_index*block_size+block_size,
            i%(self.h_index*self.w_index)%self.w_index*block_size:i%(self.h_index*self.w_index)%self.w_index*block_size+block_size,:].astype(np.float32)
            srimage = Converter.convert_bgr_to_y(self.inputs[i])
            srimage_ycbcr = cv2.resize(self.inputs[i], None, fx = self.scale, fy = self.scale, interpolation = cv2.INTER_CUBIC)
            srimage_ycbcr = Converter.convert_bgr_to_ycbcr(srimage_ycbcr)
            srimage = torch.from_numpy(srimage).to(self.device)
            srimage = srimage.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                srimage = self.model(srimage).mul(255)
            srimage = srimage.cpu().numpy().squeeze(0).squeeze(0)
            srimage = np.array([srimage, srimage_ycbcr[..., 1], srimage_ycbcr[..., 2]]).transpose(2,1,0)
            srimage = np.clip(Converter.convert_ycbcr_to_bgr(srimage), 0.0, 255.0).astype(np.uint8)
            hrimage = self.HRimages[i//(self.h_index*self.w_index)][i%(self.h_index*self.w_index)//self.w_index*block_size*self.scale:i%(self.h_index*self.w_index)//self.w_index*block_size*self.scale+block_size*self.scale,
            i%(self.h_index*self.w_index)%self.w_index*block_size*self.scale:i%(self.h_index*self.w_index)%self.w_index*block_size*self.scale+block_size*self.scale,:]
            self.psnrs[i] = cv2.PSNR(srimage, hrimage)
            if i % 30 == 0:
                print("\r"+"init {}% done!".format(int(i/(self._numbers*self.h_index*self.w_index)*100)), end="")

    def __len__(self):
        return self._numbers

    def __getitem__(self, idx):
        if self._i == self._numbers*self.h_index*self.w_index:
            raise StopIteration()
        self._i += 1
        transform = torchvision.transforms.ToTensor()
        return transform(self.inputs[idx]), 1/(1+math.e**-self.psnrs[idx])



class InferPSNR(torch.nn.Module):
    def __init__(self):
        super(InferPSNR, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=5//2),
            torch.nn.PReLU(16)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=3//2),
            torch.nn.PReLU(1),
            torch.nn.Flatten()
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.conv1:
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                torch.nn.init.zeros_(m.bias.data)
        for m in self.conv2:
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                torch.nn.init.zeros_(m.bias.data)
        for m in self.linear:
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = InferPSNR()
    d = dataset("./Best1.mp4","./GTBest1.mp4")
    dataloader = torch.utils.data.DataLoader(d, shuffle=True, batch_size=10000)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    total_loss_min = 1e+100
    max_state = None
    for epoch in range(100):
        total_loss = 0
        for image, psnr in dataloader:
            optimizer.zero_grad()
            out_psnr = model(image)
            loss = (psnr-out_psnr)**2
            loss = torch.sum(loss)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            del loss
        if total_loss < total_loss_min:
            max_state = model.state_dict()
            total_loss_min = total_loss
        if (epoch+1) % 10 == 0:
            print("epoch:{}/100 loss:{}\n".format(epoch,total_loss))
            torch.save(model.state_dict(), "./w/{}_epoch_loss{}.pth".format(epoch, total_loss))
    torch.save(max_state, "./w/best.pth")
        

