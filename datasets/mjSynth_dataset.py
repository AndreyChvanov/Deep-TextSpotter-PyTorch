import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
import random
import numpy as np
from datasets.custom_transorms import *
import re
from datasets.utils import char2index, index2char, vocab
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class MjSynthDataset(Dataset):
    def __init__(self, config, train=True, mean=None):
        self.data_dir = config.data_dir
        self.train = train
        self.mean = mean
        if train:
            self.annotation_path = os.path.join(self.data_dir, 'annotation_train.txt')
        else:
            self.annotation_path = os.path.join(self.data_dir, 'annotation_test.txt')
        self.vocab_capacity = len(vocab)
        with open(self.annotation_path, 'r') as f:
            self.img_paths = f.read().split('\n')
        self.img_paths = [item.replace('./', '').split(' ')[0] for item in self.img_paths]
        self.processing_images_paths()
        self._transform_pipeline()
        if self.mean is None:
            self.__compute_mean()

    def __compute_mean(self):
        pixel_sum = 0
        count_pixel = 0
        sum_channel_sqr = 0
        for i, path in enumerate(self.img_paths):
            if i > 2500:
                break
            pil_image = Image.open(path)
            image = np.asarray(pil_image).astype(int)
            pixel_sum += image.sum()
            sum_channel_sqr += np.power(image, 2).sum()
            count_pixel += image.shape[0] * image.shape[1] * image.shape[2]

        mean = pixel_sum / count_pixel
        ss = sum_channel_sqr - (pixel_sum ** 2 / count_pixel)
        std = np.sqrt(ss / count_pixel)
        self.mean = [mean, mean, mean]
        # self.std = [std, std, std]

    def processing_images_paths(self):
        images_paths = []
        count_images = 0
        images_paths = []
        count_images = 0
        # image_subset = random.choices(self.img_paths, k=10000)
        for path in self.img_paths:
            # try:
            #     i = Image.open(os.path.join(self.data_dir, path))
            # except:
            #     continue
            if self.train:
                if count_images > 1000000:
                    break
            if not self.train:
                if count_images > 100000:
                    break
            images_paths.append(os.path.join(self.data_dir, path))
            count_images += 1

        print('data ')
        self.img_paths = images_paths

    def _transform_pipeline(self):
        if self.train:
            self.transforms = transforms.Compose([
                VerticalResize(new_h=32),
                RandomHorizontalCrop(prob=0.0),
                RandomHorizontalResize(prob=0.0),
                HorizontalResize(max_w=500),
                RandomGaussianNoise(mean=0, std=200, prob=0.0),
            ])
        else:
            self.transforms = transforms.Compose([
                VerticalResize(new_h=32),
                HorizontalResize(max_w=500),
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        try:
            img = self.transforms(Image.open(path))
        except:
            return None, None

        leaf = path.split('/')[-1].split('.')[0]
        label = leaf.split('_')[1]
        # label = re.sub('[0-9_]', '', path.split('/')[-1].split('.')[0]).lower()
        try:
            label = [char2index[char] for char in label]
        except:
            label = []
        if len(label) == 0:
            return None, None
        normalize_img = (np.array(img) - self.mean) / 255
        normalize_img = normalize_img.mean(axis=-1)[:, :, None]
        return transforms.ToTensor()(normalize_img).permute(0, 2, 1), label


def collage_fn(batch):
    imgs = [data[0] for data in batch if data[0] is not None]
    ori_sizes = [img.shape[1] for img in imgs]
    max_len = np.max(ori_sizes)
    for i in range(len(imgs)):
        img = imgs[i]
        pad_image = torch.zeros(img.shape[0], max_len, img.shape[2])
        pad_image[:, :img.shape[1], :] = img
        imgs[i] = pad_image

    targets = [item[1] for item in batch if item[1] is not None]
    return [torch.stack(imgs), targets, ori_sizes]


if __name__ == "__main__":
    from configs.dataset_config import cfg as dataset_config

    dataset = MjSynthDataset(dataset_config)
    print(len(dataset))
    print(dataset[2][0].shape)
    dataloader = DataLoader(dataset, collate_fn=collage_fn, batch_size=32, drop_last=True)
    for batch in dataloader:
        print(batch)
