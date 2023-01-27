from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
all_transforms = [transforms.ToPILImage(),
                  transforms.Resize((256, 256)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                  ]


class DataMapper(Dataset):
    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory

        self.cloud_directory = self.dataset_directory + "cloud/"
        self.image_file_cloud = os.listdir(self.cloud_directory)

        self.label_directory = self.dataset_directory + "label/"
        self.image_file_label = os.listdir(self.label_directory)

        self.transform = transforms.Compose(all_transforms)

    def __len__(self):
        return len(self.image_file_cloud)

    def __getitem__(self, item):
        cur_cloud_image_file = self.image_file_cloud[item]
        cur_label_image_file = self.image_file_label[item]

        cur_cloud_image_path = os.path.join(self.cloud_directory, cur_cloud_image_file)
        cur_label_image_path = os.path.join(self.label_directory, cur_label_image_file)

        cur_cloud_image = np.array(Image.open(cur_cloud_image_path))
        cur_label_image = np.array(Image.open(cur_label_image_path))

        cur_cloud_image = self.transform(cur_cloud_image)
        cur_label_image = self.transform(cur_label_image)

        return cur_cloud_image, cur_label_image


# Unit-Test
if __name__ == "__main__":
    loader_cloud = DataLoader(DataMapper("thin_cloud/training/"), batch_size=5)

    for images in loader_cloud:
        for cur_image in images:
            print(cur_image.shape)
