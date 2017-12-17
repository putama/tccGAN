import time
import os
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer_Tensorboard
from util.util import create_log_path, video_writer

opt = TrainOptions().parse()

log_dir = os.path.join("logs", opt.name, create_log_path())
print("Saving log to %s" % (log_dir))

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

test_data_loader = CreateDataLoader(opt, "test")
test_dataset = test_data_loader.load_data()

model = create_model(opt)
# visualizer = Visualizer(opt)
visualizer_tb = Visualizer_Tensorboard(opt, log_dir)
total_steps = 0

# compute temporal loss
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
        if total_steps % opt.display_freq == 0:
            visualizer_tb.display_current_results(model.get_current_tensors(), epoch)
            if total_steps % opt.update_html_freq == 0:
                visualizer_tb.save_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer_tb.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer_tb.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    if epoch % opt.save_video_freq == 0:
        historydict_A = {}
        historydict_B = {}
        video_writer_A = video_writer("A", epoch, log_dir)
        video_writer_B = video_writer("B", epoch, log_dir)
        for i, data in enumerate(test_dataset):
            if historydict_A.has_key(data['A_paths'][0]) or historydict_B.has_key(data['B_paths'][0]):
                break
            video_writer_A.write(model.translateA(data['A']))
            video_writer_B.write(model.translateB(data['B']))
            historydict_A[data['A_paths'][0]] = 1
            historydict_B[data['B_paths'][0]] = 1
        video_writer_A.release()
        video_writer_B.release()
            
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
