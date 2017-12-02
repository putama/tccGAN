import numpy as np
import torch
from torch.autograd import Variable

def compute_opt_flow(imgs):
    imh = imgs.shape[2]
    imw = imgs.shape[3]
    nframe = imgs.shape[0] - 1
    if nframe == 0:
        flows = torch.FloatTensor(1, imh, imw, 2).uniform_()
    else:
        flows = torch.FloatTensor(nframe, imh, imw, 2).uniform_()
    return Variable(flows)
