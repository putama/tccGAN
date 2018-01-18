import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from tensorboardX import SummaryWriter
import torch

class Visualizer_Tensorboard():
    def __init__(self, opt, log_dir):
        # log_dir = util.create_log_path()
        self.save_path = log_dir
        self.writer = SummaryWriter(self.save_path)
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        print('create img directory %s...' % self.img_dir)
        util.mkdirs([self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals, epoch):
        idx = 0
        for label, image_tensor in visuals.items():
            img = image_tensor
            self.writer.add_image(str(label), img[[0, 1, 2], ...], idx)
            idx += 1

    def save_current_results(self, visuals_np, epoch):
        for label, image_numpy in visuals_np.items():
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            util.save_image(image_numpy, img_path)

    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        
        self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.plot_data['X'] = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)

        # print((self.plot_data['legend'], self.plot_data['Y'], self.plot_data['X']))
        for i in range(len(self.plot_data['Y'][0])):
            self.writer.add_scalar(self.plot_data['legend'][i], self.plot_data['Y'][0][i], self.plot_data['X'][0][i].item())
        self.writer.add_scalar("Loss_sum", sum(self.plot_data['Y'][0]), self.plot_data['X'][0][1].item())

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
