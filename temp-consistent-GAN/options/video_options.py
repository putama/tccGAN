from argparse import Namespace

def create_video_opt(opt):
    opt = Namespace(**vars(opt))
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.phase = "test"
    opt.dataset_mode = 'unaligned'

    return opt
