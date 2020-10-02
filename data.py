import os
from numpy import load
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DataFolder(Dataset):
    def __init__(self, trace_path, img_path):
        self.trace_path = trace_path
        self.img_path = img_path
        
        self.img_names = sorted(os.listdir(img_path))

        self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])
        print('Total %d Data.' % len(self.img_names))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        prefix = name.split('.')[0]
        suffix = '.npz'
        
        trace = load(self.trace_path + prefix + suffix)['arr_0']
        trace = trace.astype(np.float32)
        trace = torch.from_numpy(trace).view([3, 512, 512])

        img = Image.open(self.img_path + name)
        img = self.transforms(img)
        return trace, img

class DataLoader(object):
    def __init__(self, args):
        gpus = torch.cuda.device_count()
        trace_path = args['trace_path']
        img_path = args['image_path']
        batch_size = args['batch_size']
        num_workers = args['num_workers']
        train_folder = DataFolder(
            trace_path=os.path.join(trace_path, 'train'),
            img_path=os.path.join(image_path, 'train')
            )
        test_folder = DataFolder(
            trace_path=os.path.join(trace_path, 'test'),
            img_path=os.path.join(image_path, 'test')
            )
        self.train_loader = torch.utils.data.DataLoader(
            train_folder,
            batch_size=batch_size * gpus,
            num_workers=num_workers * gpus,
            shuffle=True,
            drop_last=False
            )
        self.test_loader = torch.utils.data.DataLoader(
            test_folder,
            batch_size=batch_size * gpus,
            num_workers=num_workers * gpus,
            shuffle=False,
            drop_last=False
            )
    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader