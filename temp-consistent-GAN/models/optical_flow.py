import os
import numpy as np
import torch
import torch.nn as nn
import opt
from torch.autograd import Variable
from util.util import tensor2im
from PIL import Image

def compute_opt_flow(imgs, net, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    nframe, _, imh, imw = imgs.size() # n, c, h, w
    imgs = imgs.data

    input_t1_v = imgs[0,...].unsqueeze(0)
    input_t2_v = imgs[1,...].unsqueeze(0)
    for i in range(2, imgs.size()[0]-1, 2):
        input_t1_v = torch.cat((input_t1_v, imgs[i,...].unsqueeze(0)), 0)
        input_t2_v = torch.cat((input_t2_v, imgs[i+1,...].unsqueeze(0)), 0)
    # change to n/2, 6, h, w
    input_v = Variable(torch.cat([input_t1_v, input_t2_v], 1), volatile = True)
    b, _, h, w = input_v.size()

    if use_gpu:
        input_v.cuda()
    output = net(input_v)
    
    upsampled_output = nn.functional.upsample(output, size=(h,w), mode='bilinear')
    flow_v = Variable(upsampled_output.data)
    flow_v = flow_v.permute(0, 2, 3, 1)
    assert torch.Size([nframe-1, h, w, 2]) == flow_v.size()

    if use_gpu:
        flow_v.cuda()

    return flow_v

def load_flownet(gpu_ids=[], path=None):
    use_gpu = len(gpu_ids) > 0
    if path == None:
        path = './models/opt/pretrained_model/flownets_from_caffe.pth.tar.pth'
    # Load pretrained model
    pre_model = opt.FlowNetS(batchNorm=False)

    pre_model_info = torch.load(path)
    # pre_model.load_state_dict(pre_model_info['state_dict'])
    pre_model.load_state_dict(pre_model_info)

    if torch.cuda.is_available() and use_gpu:
        pre_model.cuda(gpu_ids[0])

    # print("Load trained model ... epoch = %d" %(pre_model_info['epoch']))
    for param in pre_model.parameters():
        param.requires_grad = False
    pre_model.eval()
    
    return pre_model

def save_imgflow(img, flow, save_path=None):
    img = img[0,:,:,:].squeeze().data.cpu().numpy()
    flow = flow[0,:,:,:].squeeze().data.cpu().numpy()
    flowImg = _flow2rgb(20 * flow, max_value=20)
    if save_path == None:
        save_path = './imgs'
    
    imsave(os.path.join(save_path, "img.png"), _draw_flow(img, flow*10))
    imsave(os.path.join(save_path, "flow.png"), flowImg)
    
def _visualize(imgs, flow_map):
    background = Image.fromarray(tensor2im(imgs[0:1]))
    vals = _flow2rgb(flow_map[0].data.permute(2, 0, 1), max_value=10)
    mean = np.mean(vals)
    pixels = (vals - mean) * 255
    foreground = Image.fromarray(pixels.astype(np.uint8))
    Image.blend(foreground, background, alpha=0.5).show()

def _flow2rgb(flow_map, max_value):
    _, h, w = flow_map.shape
    rgb_map = np.ones((h,w,3)).astype(np.float32)
    normalized_flow_map = flow_map/max_value
    rgb_map[:,:,0] += normalized_flow_map[0]
    rgb_map[:,:,1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:,:,2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

def _draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    #vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = img
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis
