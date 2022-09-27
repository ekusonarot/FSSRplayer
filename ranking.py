import os
import sys
import torch
import cv2
import math
from torchviz import make_dot
sys.path.append(os.path.abspath("."))
from models import FSRCNN

class RankSR(torch.nn.Module):
  def __init__(self, scale_factor=4, patch_size=20, device="cpu"):
    super(RankSR, self).__init__()
    self.ranking = Ranking(device)
    self.scale_factor = scale_factor
    self.patch_size = patch_size
    self.downsample = torch.nn.Upsample(size=(20,20))
    self.net1 = FSRCNN(scale_factor=self.scale_factor).to(device)
    state_dict = self.net1.state_dict()
    for n, p in torch.load("./weights/fsrcnn_x4.pth", map_location=lambda storage, loc: storage).items():
      if n in state_dict.keys():
          state_dict[n].copy_(p)
      else:
          raise KeyError(n)
    self.net1.eval()
    self.net2 = torch.nn.Upsample(scale_factor=(self.scale_factor, self.scale_factor))
    for i in self.net1.parameters():
      i.requires_grad=False

  def forward(self, x, is_train=True):
    if is_train:
      rank = self.ranking(x)
      with torch.no_grad():
        output1 = self.net1(x).detach()
        output2 = self.net2(x).detach()
      return rank, output1, output2
    else:
      rank = self.ranking(x)
      return rank

class Ranking(torch.nn.Module):
  def __init__(self, device="cuda:0"):
    super(Ranking, self).__init__()
    self.downsample = torch.nn.Upsample(size=(20,20))
    self.conv2d_1 = torch.nn.Sequential(
      torch.nn.Conv2d(3,3,3, padding=3//2),
      torch.nn.LeakyReLU(),
      torch.nn.AvgPool2d(2, stride=2),
      torch.nn.Conv2d(3,1,3, padding=3//2),
      torch.nn.LeakyReLU(),
      torch.nn.AvgPool2d(2, stride=2),
    )
    self.conv2d_2 = torch.nn.Sequential(
      torch.nn.Conv2d(1,1,4, stride=5),
      torch.nn.Tanhshrink(),
      torch.nn.Flatten(),
    )
    self._initialize_weights()

  def _initialize_weights(self):
    for m in self.conv2d_1:
      if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.3)
        torch.nn.init.zeros_(m.bias.data)
    for m in self.conv2d_2:
      if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.3)
        torch.nn.init.zeros_(m.bias.data)

  def forward(self, x):
    x = self.downsample(x)
    x = self.conv2d_1(x)
    x = self.conv2d_2(x)
    return x

class CustomLoss(torch.nn.Module):
  def __init__(self, device="cuda:0"):
    super(CustomLoss, self).__init__()
    self.device=device

  def forward(self, rank, output1, output2, targets):
    trans_mat = torch.ones(1, rank.size(0)).to(self.device)

    diff1 = torch.abs(output1-targets)
    diff2 = torch.abs(output2-targets)
    sum1 = torch.sum(diff1, dim=(1,2,3)).unsqueeze(1)
    sum2 = torch.sum(diff2, dim=(1,2,3)).unsqueeze(1)

    diff = sum2 - sum1
    true_rank = torch.matmul(diff, trans_mat)
    true_rank = true_rank.transpose(0, 1) - true_rank
    true_rank = torch.sigmoid(true_rank*1_000_000)

    pred_rank = torch.matmul(rank, trans_mat)
    pred_rank = pred_rank.transpose(0, 1) - pred_rank
    pred_rank = torch.sigmoid(pred_rank)
    
    loss = torch.sum((true_rank-pred_rank)**2)/pred_rank.numel()
    return loss

class ToPatches:
  def __init__(self, size, padding=(0, 0, 0, 0), mode="constant"):
    self.size = size
    self.padding = padding
    self.mode = mode

  def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
    wid = imgs.size(2)//self.size[0]
    hei = imgs.size(3)//self.size[1]
    xp = self.padding[2]+self.padding[3]
    yp = self.padding[0]+self.padding[1]
    out = torch.zeros((imgs.size(0)*wid*hei, imgs.size(1), self.size[0]+xp, self.size[1]+xp))
    imgs_pad = torch.nn.functional.pad(imgs, self.padding, self.mode)
    i = 0
    for b in range(imgs.size(0)):
      for x in range(0, imgs.size(2), self.size[0]):
        for y in range(0, imgs.size(3), self.size[1]):
          t = imgs_pad[b,:,x:x+xp+self.size[0],y:y+yp+self.size[1]].unsqueeze(0)
          out[i,:,:,:]=t
          i += 1
    return out
    
if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = RankSR(device=device).to(device)
  x = torch.randn(1, 1, 180, 320).to(device)
  y = model(x)
  image = make_dot(y, params=dict(model.named_parameters()))
  image.format = "png"
  image.render("NeuralNet")