import numpy as np
import collections
import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    elif opt.dataset_mode == 'triplet':
        from data.triplet_dataset import TripletDataset
        dataset = TripletDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        if opt.dataset_mode == 'triplet':
            self.dataloader.collate_fn = custom_collate

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data

def custom_collate(batch):
    error_msg = "batch must contain tensors; found {}"
    collatedbatch = {}
    collatedbatch['A'] = torch.cat([batch[i]['A'] for i in range(len(batch))], 0)
    collatedbatch['B'] = torch.cat([batch[i]['B'] for i in range(len(batch))], 0)

    flowpathsA = [batch[i]['A_paths'][2] for i in range(len(batch))]
    flowpathsB = [batch[i]['B_paths'][2] for i in range(len(batch))]
    flowsA = map(lambda x: torch.FloatTensor(load_flo(x)), flowpathsA)
    flowsB = map(lambda x: torch.FloatTensor(load_flo(x)), flowpathsB)
    flowsAtensor = torch.stack(flowsA, 0)
    flowsBtensor = torch.stack(flowsB, 0)

    collatedbatch['A_flows'] = flowsAtensor
    collatedbatch['B_flows'] = flowsBtensor

    collatedbatch['A_paths'] = [batch[i]['A_paths'] for i in range(len(batch))]
    collatedbatch['B_paths'] = [batch[i]['B_paths'] for i in range(len(batch))]

    return collatedbatch

def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D