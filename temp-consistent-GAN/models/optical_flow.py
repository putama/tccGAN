import os
import numpy as np
import torch
import torch.nn as nn
import opt
from torch.autograd import Variable

def compute_opt_flow(imgs, net):
    print(imgs.size())
    imh, imw = imgs.size()[2:4]
    nframe = imgs.size()[0] - 1

    if torch.cuda.is_available():
        input_var = Variable(torch.cat(imgs[0:2,...].data,0).unsqueeze(0).cuda(), volatile=True)
    else:
        input_var = Variable(torch.cat(imgs[0:2, ...].data, 0).unsqueeze(0), volatile=True)

    for i in range(2,nframe+1):
        if torch.cuda.is_available():
            input_var.stack(Variable(torch.cat(imgs[i:i+2,...].data,0).unsqueeze(0).cuda(), volatile=True))
        else:
            input_var.stack(Variable(torch.cat(imgs[i:i + 2, ...].data, 0).unsqueeze(0), volatile=True))

    b, _, h, w = input_var.size()
    output = net(input_var)
    upsampled_output = nn.functional.upsample(output, size=(h,w), mode='bilinear')
    flow = Variable(upsampled_output.data[0])

    if torch.cuda.is_available():
        flow.cuda()

    flow = flow.permute(1, 2, 0).unsqueeze(0)
    print(flow.size())
    b_out, h_out, w_out, c_out = flow.size()
    assert nframe == b_out and h == h_out and w == w_out and c_out == 2
    
    return flow

def load_flownet(path=None):
    if path == None:
        path = './models/opt/pretrained_model/flownets_from_caffe.pth.tar.pth'
    # Load pretrained model
    pre_model = opt.FlowNetS(batchNorm=False)

    pre_model_info = torch.load(path)
    # pre_model.load_state_dict(pre_model_info['state_dict'])
    pre_model.load_state_dict(pre_model_info)

    if torch.cuda.is_available():
        pre_model.cuda()

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
