from options.test_options import TestOptions
import os
import numpy as np
from options.video_options import create_video_opt
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.util import video_writer

opt = TestOptions().parse()

# Base parameters
opt.model = "cycle_gan"
AtoB = opt.which_direction == 'AtoB'
epoch = opt.which_epoch

mode = ["test","eval"]
log_dir = os.path.join("logs", opt.name, "evaluate")

# Create dataset
opt_eval = create_video_opt(opt)
opt_eval.phase = "eval"
eval_data_loader = CreateDataLoader(opt_eval)
eval_dataset = eval_data_loader.load_data()

opt_test = create_video_opt(opt)
test_data_loader = CreateDataLoader(opt_test)
test_dataset = test_data_loader.load_data()

# Create model
model = create_model(opt_test)

for ind, dataset in enumerate([test_dataset, eval_dataset]):
    hdict_A = {}
    hdict_B = {}
    video_writer_A = video_writer("A", epoch, log_dir, mode[ind])
    video_writer_B = video_writer("B", epoch, log_dir, mode[ind])
    video_writer_A_org = video_writer("A", epoch, log_dir, mode[ind]+"_original")
    video_writer_B_org = video_writer("B", epoch, log_dir, mode[ind]+"_original")

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.forward()
        if hdict_A.has_key(data['A_paths'][0]) and hdict_B.has_key(data['B_paths'][0]):
            break
        video_writer_A.write(model.translateA())
        video_writer_B.write(model.translateB())
        
        real_A, real_B = model.get_realAB()
        video_writer_A_org.write(real_A)
        video_writer_B_org.write(real_B)
        
        hdict_A[data['A_paths'][0]] = 1
        hdict_B[data['B_paths'][0]] = 1
        
    video_writer_A.release()
    video_writer_B.release()
    print("Saving " + mode[ind] + " video")
    video_writer_A_org.release()
    video_writer_B_org.release()
