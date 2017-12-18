from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.util import save_image
from util.util import tensor2im
from util.util import mkdirreplace
import cv2
import numpy as np

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.resize_or_crop = "scale_width" # no random crop
opt.dataset_mode = "unaligned" # unaligned dataset
opt.phase = "test"
opt.model = "cycle_gan"
opt.no_dropout = True
AtoB = opt.which_direction == 'AtoB'

data_loader = CreateDataLoader(opt, "test")
dataset = data_loader.load_data()
model = create_model(opt)

frame_width = 256
frame_height = 256
out = cv2.VideoWriter(opt.name+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width * 2,frame_height))

historydict = {}
for i, data in enumerate(dataset):
    if AtoB:
        if historydict.has_key(data['A_paths'][0]):
            break
        model.set_input(data)
        im_fake = model.translateA()
        historydict[data['A_paths'][0]] = 1
	im_fake = np.concatenate((tensor2im(data['A'][:, [2, 1, 0], ...]), im_fake), axis=1)
    else:
        if historydict.has_key(data['B_paths'][0]):
            break
        model.set_input(data)
        im_fake = model.translateB()
        historydict[data['B_paths'][0]] = 1
	im_fake = np.concatenate((tensor2im(data['B'][:, [2, 1, 0], ...]), im_fake), axis=1)
    out.write(im_fake)
out.release()
