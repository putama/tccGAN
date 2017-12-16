import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import PIL
import random
import torch

class VideoDataset(BaseDataset):

    def initialize(self, opt):
        default_offset = 1
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'video_A')
        self.dir_B = os.path.join(opt.dataroot, 'video_B')

        self.A_video_tuples = self.aggregate_dataset(self.dir_A, default_offset)
        self.B_video_tuples = self.aggregate_dataset(self.dir_B, default_offset)

        self.A_size = len(self.A_video_tuples)
        self.B_size = len(self.B_video_tuples)

        self.transform = get_transform(opt)

    # A_paths and B_paths are tuples with:
    # 0 - first image path
    # 1 - second image path
    # 2 - optical flow path

    def __getitem__(self, index):
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)
        A_video_tuple = self.A_video_tuples[index_A]
        B_video_tuple = self.B_video_tuples[index_B]

        A_tuple = A_video_tuple[random.randint(0, len(A_video_tuple) - 1)]
        B_tuple = B_video_tuple[random.randint(0, len(B_video_tuple) - 1)]

        A1_img = Image.open(A_tuple[0]).convert('RGB')
        A2_img = Image.open(A_tuple[1]).convert('RGB')
        B1_img = Image.open(B_tuple[0]).convert('RGB')
        B2_img = Image.open(B_tuple[1]).convert('RGB')

        A1 = self.transform(A1_img)
        A2 = self.transform(A2_img)
        B1 = self.transform(B1_img)
        B2 = self.transform(B2_img)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp1 = A1[0, ...] * 0.299 + A1[1, ...] * 0.587 + A1[2, ...] * 0.114
            A1 = tmp1.unsqueeze(0)
            tmp2 = A2[0, ...] * 0.299 + A2[1, ...] * 0.587 + A2[2, ...] * 0.114
            A2 = tmp2.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp1 = B1[0, ...] * 0.299 + B1[1, ...] * 0.587 + B1[2, ...] * 0.114
            B1 = tmp1.unsqueeze(0)
            tmp2 = B2[0, ...] * 0.299 + B2[1, ...] * 0.587 + B2[2, ...] * 0.114
            B2 = tmp2.unsqueeze(0)

        # if this messes up, it's possible these aren't concat'd correctly:
        A = torch.cat((A1.unsqueeze(0), A2.unsqueeze(0)), 0)
        B = torch.cat((B1.unsqueeze(0), B2.unsqueeze(0)), 0)

        return {'A': A, 'B': B,
                'A_paths': A_tuple, 'B_paths': B_tuple}

    # did some testing of this already locally, I think this is good to go
    # won't behave well in certain edge cases, like when there are no flows in a directory
    def aggregate_dataset(self, dir, offset):
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        id_length = 5
        tuples = []
        root = dir
        for v in os.listdir(root):
            video_tuples = []
            rootv = os.path.join(root, v)
            if os.path.isdir(rootv):
                idToImage = {}
                for filename in os.listdir(rootv):
                    ext = filename.rfind('.')
                    file_id = int(filename[ext-id_length:ext])
                    idToImage[file_id] = os.path.join(rootv, filename)
                for key in idToImage:
                    if key in idToImage and (key + offset) in idToImage:
                        video_tuples.append((idToImage[key], idToImage[key + offset]))
                tuples.append(video_tuples)
        return tuples

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'VideoDataset'