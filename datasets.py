import json
from PIL import Image
from pathlib import Path
import os

from torch.utils.data import Dataset
from glob import glob

class VehicleDataset(Dataset):
    def __init__(self, root, transform, mode='train'):
        self.data_root = root
        self.mode = mode
        self.transform = transform
        self.images = sorted(glob(self.data_root + '/*.jpg'))
        if mode == 'train':
            self.annotations = sorted(glob(self.data_root + '/*.json'))

    def __len__(self):
        if self.mode == 'train':
            assert len(self.images) == len(
                self.annotations), "# of image files and # of json files do not match"
        print('hello : ', len(self.images))
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image)
        image = self.transform(image)

        if self.mode == 'train':
            annotation = self.annotations[index]
            label = self.get_gt(annotation)
            return image, label
        else:
            return image

    def get_gt(self, json_file):
        label_dict = {
            'motorcycle': 0,
            'concrete': 1,
            'bus': 2,
            'benz': 3,
            'suv': 4
        }

        json_file = Path(json_file)
        with open(json_file, 'r') as f:
            annotation = json.load(f)
        gt_class = label_dict[(annotation['label'])]

        return gt_class