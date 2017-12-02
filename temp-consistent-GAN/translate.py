import os
import torch
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.util import save_image

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 2  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
AtoB = opt.which_direction == 'AtoB'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
# model = create_model(opt)

for i, data in enumerate(dataset):
    if i == 0:
        prevframe = data['A' if AtoB else 'B'][1:2]
    else:
        nextframe = data['A' if AtoB else 'B'][0:1]
        data['A' if AtoB else 'B'] = torch.cat((prevframe, nextframe), 0)

        # translate the data using trained model here
        

        # update prevframe from this batch
        prevframe = data['A' if AtoB else 'B'][1:2]
    print 'done'
    # model.set_input(data)
    # model.translate()