from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.util import save_image
from util.util import tensor2im
from util.util import mkdirreplace
import cv2

rootpath = "imgs/results/"
mkdirreplace(rootpath)

frame_width = 256
frame_height = 256
out = cv2.VideoWriter('outpylatest.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
AtoB = opt.which_direction == 'AtoB'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

historydict = {}
for i, data in enumerate(dataset):
    if AtoB:
        if historydict.has_key(data['A_paths'][0]):
            break
        filepath = rootpath + "fake_B_{}.jpg".format(str(i))
        im_fake = model.translateA(data['A'])
        historydict[data['A_paths'][0]] = 1
    else:
        if historydict.has_key(data['B_paths'][0]):
            break
        filepath = rootpath + "fake_A_{}.jpg".format(str(i))
        im_fake = model.translateB(data['B'][0])
    save_image(im_fake, filepath)
    print "saved to " + filepath
    out.write(im_fake)
out.release()