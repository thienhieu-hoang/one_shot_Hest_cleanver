# download at: https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset

#%%
import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :] # H x W x dim (600, 1200, 3)
        target_image = image[:, 600:, :] # H x W x Dim (600,1200,3)

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

#%%
if __name__ == "__main__":
    dataset = MapDataset(FILE_PATH+"/data/maps/maps/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, FILE_PATH+"/results/dataset_test/x.png")
        save_image(y, FILE_PATH+"/results/dataset_test/y.png")
        import sys

        sys.exit()
