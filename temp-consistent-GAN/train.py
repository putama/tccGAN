import time
<<<<<<< HEAD
=======
import os
>>>>>>> 0cbd6819bb3f8d7b860c0e27b9c3a802eddcb150
import sys
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from models.cycle_gan_3d import CycleGAN3dModel
from util.visualizer import Visualizer_Tensorboard
from util.util import create_log_path, video_writer
from options.video_options import create_video_opt

opt = TrainOptions().parse()

log_dir = os.path.join("logs", opt.name, create_log_path())
print("Saving log to %s" % (log_dir))

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('# training images = %d' % dataset_size)

test_data_loader = CreateDataLoader(create_video_opt(opt))
test_dataset = test_data_loader.load_data()

if opt.model == "3d_cycle_gan":
    print("Using 3D conv for cycle_gan")
    model = CycleGAN3dModel()
    model.initialize(opt)
else:
    model = create_model(opt)
    
visualizer = Visualizer_Tensorboard(opt, log_dir)
total_steps = 0

# compute temporal loss
if opt.dataset_mode == "video":
    model.video_mode = True

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    model.train()
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        try:
            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_tensors(), epoch)
                if total_steps % opt.update_html_freq == 0:
                    visualizer.save_current_results(model.get_current_visuals(), epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
        except:
            print "Unexpected error:", sys.exc_info()[0]

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
        
    if epoch % opt.save_video_freq == 0:
        print('saving video ...')
        hdict_A = {}
        hdict_B = {}
        video_writer_A = video_writer("A", epoch, log_dir)
        video_writer_B = video_writer("B", epoch, log_dir)
        for i, data in enumerate(test_dataset):
            if hdict_A.has_key(data['A_paths'][0]) and hdict_B.has_key(data['B_paths'][0]):
                break
            video_writer_A.write(model.translateA(data['A']))
            video_writer_B.write(model.translateB(data['B']))
            hdict_A[data['A_paths'][0]] = 1
            hdict_B[data['B_paths'][0]] = 1
        video_writer_A.release()
        video_writer_B.release()
            
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
