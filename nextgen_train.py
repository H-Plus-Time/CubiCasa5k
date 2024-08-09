import typer
from pathlib import Path
import lightning as L
from lightning.pytorch.tuner import Tuner
import torch
from enum import Enum
from typing import Optional, Any, Literal
from floortrans.loaders import FloorplanSVG, FloorplanSVGSample
from floortrans.loaders.augmentations import (
    RandomCropToSizeTorch,
    ResizePaddedTorch,
    Compose,
    DictToTensor,
    ColorJitterTorch,
    RandomRotations,
)
from floortrans.losses import UncertaintyLoss
from torchvision.transforms import RandomChoice
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os


from floortrans.models import get_model
from floortrans.models.hg_furukawa_original import hg_furukawa_original

class ModelArchitecture(str, Enum):
    HG_FURUKAWA_ORIGINAL = "hg_furukawa_original"


class OptimizerSelection(str, Enum):
    ADAM_PATIENCE_PREVIOUS_BEST = "adam-patience-previous-best"
    ADAM_PATIENCE = "adam-patience"
    ADAM_SCHEDULER = "adam-scheduler"
    SGD = "sgd"

class LitModel(L.LightningModule):
    def __init__(
        self,
        internal_model: hg_furukawa_original,
        criterion,
        optimizer: OptimizerSelection,
        optimizer_options: dict,
        n_epoch: int,
        l_rate: float = 1e-3
    ):
        super().__init__()
        self.model = internal_model
        # TODO: align this with what's in the original train.py
        self.criterion = criterion
        self._optimizer = optimizer
        self._optimizer_options = optimizer_options
        self.n_epoch = n_epoch
        self.l_rate = l_rate

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)
    
    def training_step(self, batch: FloorplanSVGSample, batch_idx: int):
        output = self.model(batch.image)
        loss = self.criterion(output, batch.label)
        return loss

    def validation_step(self, batch: FloorplanSVGSample, batch_idx: int):
        output = self.model(batch.image)
        labels_val = F.interpolate(
            batch.label, size=output.shape[2:], mode="bilinear", align_corners=False
        )
        loss = self.criterion(output, labels_val)
        return loss
    def predict_step(self, batch: FloorplanSVGSample, batch_idx: int):
        output = self.model(batch.image)
        return output
    
    def test_step(self, batch: FloorplanSVGSample, batch_idx: int):
        print(batch, batch.image)
        output = self.model(batch.image)
        return output

    
    def transfer_batch_to_device(self, batch: FloorplanSVGSample | Any, device: torch.device, dataloader_idx: int):
        if isinstance(batch, FloorplanSVGSample):
            batch.image = batch.image.to(device)
            batch.label = batch.label.to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch
    
    def configure_optimizers(self):
        params = [{'params': self.model.parameters(), 'lr': self.l_rate},
                {'params': self.criterion.parameters(), 'lr': self.l_rate}]
        if self._optimizer == OptimizerSelection.ADAM_PATIENCE:
            optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self._optimizer_options['patience'], factor=0.5)
        elif self._optimizer == OptimizerSelection.ADAM_PATIENCE_PREVIOUS_BEST:
            optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
            scheduler = None
        elif self._optimizer == OptimizerSelection.SGD:
            def lr_drop(epoch):
                return (1 - epoch/self.n_epoch)**0.9
            optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=10**-4, nesterov=True)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_drop)
        elif self._optimizer == OptimizerSelection.ADAM_SCHEDULER:
            def lr_drop(epoch):
                return 0.5 ** np.floor(epoch / self._optimizer_options['l_rate_drop'])
            optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_drop)
        if scheduler is not None:
            extras = dict(lr_scheduler=scheduler)
        else:
            extras = {}
        return dict(
            optimizer=optimizer,
        ) | extras

class LitDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, scale: int, image_size: int = 256, batch_size: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.scale = scale
        self.image_size = image_size
    def setup(self, stage: str):
        pass
        

    def train_dataloader(self):
        augmentations = define_augmentations(self.scale, self.image_size)
        self.train_set = FloorplanSVG(
            self.data_dir, "train.txt", format="lmdb", augmentations=augmentations
        )
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size | self.hparams.batch_size,
            num_workers=os.cpu_count() // 2,
            shuffle=True,
            pin_memory=True,
            collate_fn=custom_collate,
        )
        return train_loader
    def val_dataloader(self):
        self.val_set = FloorplanSVG(
            self.data_dir, "val.txt", format="lmdb", augmentations=DictToTensor()
        )
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=os.cpu_count() // 2,
            pin_memory=True,
            collate_fn=custom_collate,
        )
    def test_dataloader(self):
        test_set = FloorplanSVG(
            self.data_dir, "test.txt", format="lmdb", augmentations=DictToTensor()
        )
        return DataLoader(
            test_set,
            batch_size=1,
            num_workers=0,
            collate_fn=custom_collate,
        )
    

def custom_collate(input_values: tuple[FloorplanSVGSample, ...], *rest):
    return FloorplanSVGSample(
        image=torch.stack(tuple(x.image for x in input_values)),
        label=torch.stack(tuple(x.label for x in input_values)),
    )


def define_augmentations(scale: Optional[float], image_size: int):
    if scale:
        return Compose(
            [
                RandomChoice(
                    [
                        RandomCropToSizeTorch(
                            data_format="dict", size=(image_size, image_size)
                        ),
                        ResizePaddedTorch(
                            (0, 0),
                            data_format="dict",
                            size=(image_size, image_size),
                        ),
                    ]
                ),
                RandomRotations(format="cubi"),
                DictToTensor(),
                ColorJitterTorch(),
            ]
        )
    else:
        return Compose(
            [
                RandomCropToSizeTorch(
                    data_format="dict", size=(image_size, image_size)
                ),
                RandomRotations(format="cubi"),
                DictToTensor(),
                ColorJitterTorch(),
            ]
        )


def define_model(
    arch: ModelArchitecture,
    n_classes: int = 44,
    input_slice: Optional[list[int]]=[21, 12, 11],
    furukawa_weights: Optional[Any] = None,
):
    if arch == ModelArchitecture.HG_FURUKAWA_ORIGINAL:
        from torch import nn
        model = get_model(arch, 51)
        criterion = UncertaintyLoss(input_slice=input_slice)
        if furukawa_weights:
            checkpoint = torch.load(furukawa_weights)
            model.load_state_dict(checkpoint["model_state"])
            criterion.load_state_dict(checkpoint["criterion_state"])

        model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=4, stride=4
        )
        for m in [model.conv4_, model.upsample]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)
    else:
        model = get_model(arch, n_classes)
        criterion = UncertaintyLoss(input_slice=input_slice)
    return model, criterion

def main(
    scale: Optional[float] = None,
    data_path: Path = Path("data/cubicasa5k/"),
    batch_size: Optional[int] = 26,
    image_size: Optional[int] = 256,
    n_classes: Optional[int] = 44,
    arch: Optional[ModelArchitecture] = ModelArchitecture.HG_FURUKAWA_ORIGINAL,
    optimizer: OptimizerSelection = OptimizerSelection.ADAM_PATIENCE_PREVIOUS_BEST,
    n_epoch: Optional[int] = 1_000,
    l_rate: Optional[float] = 1e-3,
    auto_resume: Optional[bool] = False,
    debug: Optional[bool] = False,
):

    # Unknown constants
    input_slice = [21, 12, 11]
    if auto_resume:
        # Default to loading from the last known run
        all_ckpt_paths = list(Path('lightning_logs').glob('**/*.ckpt'))
        latest_ckpt = sorted(all_ckpt_paths, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        print(latest_ckpt)
        fit_extras = dict(ckpt_path=latest_ckpt)
    else:
        fit_extras = {}
    trainer = L.Trainer(
        max_epochs=n_epoch, precision='16-mixed',
        enable_checkpointing=True,
        enable_model_summary=True,
        reload_dataloaders_every_n_epochs=10,
        check_val_every_n_epoch=2
    )
    
    internal_model, criterion = define_model(arch, n_classes, input_slice)
    model = LitModel(
        internal_model, criterion, optimizer, {}, n_epoch=n_epoch, l_rate=l_rate,
    )
    datamodule = LitDataModule(data_path, scale, image_size, batch_size=batch_size)

    # tuner = Tuner(trainer)
    # guessed_batch_size = tuner.scale_batch_size(model, datamodule=datamodule, mode="binsearch")
    # print(guessed_batch_size)
    trainer.fit(model, datamodule=datamodule, **fit_extras)

if __name__ == "__main__":
    typer.run(main)
