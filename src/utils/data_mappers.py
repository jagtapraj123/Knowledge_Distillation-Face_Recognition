import os
import numpy as np
import pandas as pd

from utils.preprocessing import Preprocessor

import torch
from torch.utils.data import Dataset


class DatasetMapper(Dataset):
    """
    Dataset Mapper class to get image and label given an index.

    The class is written using torch parent class 'Dataset' for parallelizing and prefetching to accelerate training

    -----------
    Attributes:
    -----------
    - root_dir: str
        - directory path where images are stored

    - image_data_file: str
        - path to csv file that stores image names and labels

    - preprocessor: Instance of subclass of utils.Preprocessor
        - stores preprocessor with 'get' function that returns processed image given path to image and transformations

    - teacher_func: Function
        - Function to create label/feature-values given an image (for semi-supervised learning)
    """

    def __init__(
        self,
        root_dir,
        image_data_file,
        preprocessor: Preprocessor,
        teacher_func=None,
        pretrain_size=1,
        augment=False,
        **kwargs
    ):
        """
        Init for DatasetMapper

        -----
        Args:
        -----
        - root_dir: str
            - directory path where images are stored

        - image_data_file: str
            - path to csv file that stores image names and labels

        - preprocessor: Instance of subclass of utils.Preprocessor
            - stores preprocessor with 'get' function that returns processed image given path to image and transformations

        - teacher_func: Function
            - Function to create label/feature-values given an image (for semi-supervised learning)
        """

        self.root_dir = root_dir
        self.image_data_file = pd.read_csv(image_data_file)
        self.num_classes = len(self.image_data_file["label"].unique())
        self.preprocessor = preprocessor
        self.teacher_func = teacher_func
        self.pretrain_size = pretrain_size
        self.augment = augment

    def __len__(self):
        """
        Function to get size of dataset
        """

        if self.teacher_func is None:
            return self.image_data_file.shape[0]

        return self.pretrain_size

    def __getitem__(self, idx):
        """
        Mapper function to get processed image and label given an index

        -----
        Args:
        -----
        - idx: int (python int / numpy.int / torch.int)
            - index of an image
            - idx >= 0 and idx < self.__len__()
        """

        # Convert to python int
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get final image path from image data csv file
        img_name = os.path.join(self.root_dir, self.image_data_file.iloc[idx, 0])

        # Get processed image from preprocessor given image path
        if self.augment:
            image = self.preprocessor.get(
                self.preprocessor.make_random_combinations(
                    1,
                    p_transformations={
                        "rotate": 0.5,
                        "scale": 0.5,
                        "flip": 0.5,
                        "gaussian_blur": 0.1,
                        "color_jitter": 0,
                        "random_erasing": 0,
                    },
                )[0],
                image_path=img_name,
                color_jitter=None,
                rotate=np.random.randint(0, 45),
                scale=np.random.uniform(0.7, 1),
                flip="h",
                gaussian_blur=1,
                is_url=False,
            )
        else:
            image = self.preprocessor.get("", img_name, is_url=False)

        # Get image label/feature-values from teacher function (if specified)
        if self.teacher_func is not None:
            img_label = self.teacher_func(x=image)
        else:
            # If not specified, get label/feature-values from image data csv file
            # img_label = F.one_hot(torch.LongTensor([self.image_data_file.iloc[idx, 1]]), num_classes=self.num_classes)
            img_label = self.image_data_file.iloc[idx, 1]
        # print(image.shape, flush=True)
        return image, img_label


class DatasetGen(Dataset):
    def __init__(
        self,
        url,
        preprocessor: Preprocessor,
        teacher_func=None,
        pretrain_size=1,
        augment=False,
        **kwargs
    ) -> None:

        self.url = url
        self.preprocessor = preprocessor
        self.teacher_func = teacher_func
        self.pretrain_size = pretrain_size
        self.augment = augment

    def __len__(self):
        return self.pretrain_size

    def __getitem__(self, idx):
        if self.augment:
            image = self.preprocessor.get(
                self.preprocessor.make_random_combinations(
                    1,
                    p_transformations={
                        "rotate": 0.5,
                        "scale": 0.5,
                        "flip": 0.5,
                        "gaussian_blur": 0.1,
                        "color_jitter": 0,
                        "random_erasing": 0,
                    },
                )[0],
                image_path=self.url,
                is_url=True,
                color_jitter=None,
                rotate=np.random.randint(0, 45),
                scale=np.random.uniform(0.7, 1),
                flip="h",
                gaussian_blur=1,
            )
        else:
            image = self.preprocessor.get("", image_path=self.url, is_url=True)

        # Get image label/feature-values from teacher function (if specified)

        img_features = self.teacher_func(x=image)
        img_features = torch.squeeze(img_features, dim=0)

        # print(image.shape, flush=True)
        return image, img_features
