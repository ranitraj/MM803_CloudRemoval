from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

all_transforms = [transforms.ToPILImage(),
                  transforms.Resize((256, 256)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                ]

class DataMapper(Dataset):
    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory
        self.image_files = os.listdir(self.dataset_directory)
        print(self.image_files)
        self.transform = transforms.Compose(all_transforms)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        cur_image_file = self.image_files[item]
        cur_image_path = os.path.join(self.dataset_directory, cur_image_file)
        cur_image = np.array(Image.open(cur_image_path))
        cur_image = self.transform(cur_image)
        return cur_image


# Unit-Test
if __name__ == "__main__":
    # TODO: To be changed into proper structure
    #cloud = DataMapper("thin_cloud/cloud/")
    #label = DataMapper("thin_cloud/label/")

    #loader_cloud = DataLoader(cloud, batch_size=5)
    #loader_label = DataLoader(label, batch_size=5)

    loader_cloud = DataLoader(DataMapper("thin_cloud/cloud/"), batch_size=5)
    loader_label = DataLoader(DataMapper("thin_cloud/label/"), batch_size=5)

    for x in loader_cloud:
        print(x.shape)
    for x in loader_label:
        print(x.shape)
