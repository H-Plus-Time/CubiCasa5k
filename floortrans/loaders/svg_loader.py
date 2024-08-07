import lmdb
import pickle
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from numpy import typing as npt
from typing import Literal, Optional
from numpy import genfromtxt
from floortrans.loaders.house import House
import blosc2
from pydantic import BaseModel, ConfigDict
from pathlib import Path

class FloorplanSVGSample(BaseModel):
    image: torch.Tensor
    label: torch.Tensor
    folder: Optional[str] = None
    heatmaps: Optional[dict] = None
    scale: Optional[float] = None
    model_config=ConfigDict(arbitrary_types_allowed=True)

class FloorplanSVG(Dataset):
    def __init__(self, data_folder: Path, data_file: str, is_transform=True,
                 augmentations=None, img_norm=True, format='txt',
                 original_size=False, lmdb_folder='cubi_lmdb/'):
        self.img_norm = img_norm
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.get_data = None
        self.original_size = original_size
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'

        if format == 'txt':
            self.get_data = self.get_txt
        if format == 'lmdb':
            self.lmdb = lmdb.open(str(data_folder / lmdb_folder), readonly=True,
                                  max_readers=8, lock=False,
                                  readahead=True, meminit=False)
            self.get_data = self.get_lmdb
            self.is_transform = False

        self.data_folder = data_folder
        # Load txt file to list
        self.folders = genfromtxt(data_folder / data_file, dtype=np.str_)

    def __len__(self):
        """__len__"""
        return len(self.folders)

    def __getitem__(self, index: int):
        assert self.get_data is not None
        sample = self.get_data(index)
        if self.augmentations is not None:
            sample = self.augmentations(sample)
            
        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def get_txt(self, index: int):
        fplan = cv2.imread(self.data_folder / self.folders[index] / self.image_file_name)
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
        height, width, nchannel = fplan.shape
        fplan = np.moveaxis(fplan, -1, 0)

        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder / self.folders[index] / self.svg_file_name, height, width)
        # Combining them to one numpy tensor
        label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))
        heatmaps = house.get_heatmap_dict()
        coef_width = 1
        if self.original_size:
            fplan = cv2.imread(self.data_folder / self.folders[index] / self.org_image_file_name)
            fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
            height_org, width_org, nchannel = fplan.shape
            fplan = np.moveaxis(fplan, -1, 0)
            label = label.unsqueeze(0)
            label = torch.nn.functional.interpolate(label,
                                                    size=(height_org, width_org),
                                                    mode='nearest')
            label = label.squeeze(0)

            coef_height = float(height_org) / float(height)
            coef_width = float(width_org) / float(width)
            for key, value in heatmaps.items():
                heatmaps[key] = [(int(round(x*coef_width)), int(round(y*coef_height))) for x, y in value]

        img = torch.tensor(fplan.astype(np.float32))

        return FloorplanSVGSample(
            image=img, label=label,
            folder=self.folders[index],
            heatmaps=heatmaps,
            scale=coef_width
        )

    def get_lmdb(self, index: int):
        key = self.folders[index].encode()
        with self.lmdb.begin(write=False) as f:
            data = f.get(key)

        sample = pickle.loads(data)
        # unpack the two arrays
        return FloorplanSVGSample(
            image=blosc2.unpack_tensor(sample["image"]),
            label=blosc2.unpack_tensor(sample["label"]),
            folder=sample['folder'],
            heatmaps=sample['heatmaps'],
            scale=sample['scale']
        )

    def transform(self, sample: FloorplanSVGSample):
        fplan = sample.image
        # Normalization values to range -1 and 1
        fplan = 2 * (fplan / 255.0) - 1

        sample.image = fplan

        return sample
