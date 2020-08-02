# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics import DiceCoefficient


class TrainingModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: callable,
        init_lr: float = 1e-3,
        metrics: dict = {'dice_coef': DiceCoefficient()},
        seed: int = None
    ):
        super(TrainingModule, self).__init__()
        self.model = model
        self.loss = loss
        self.init_lr = init_lr
        self.metrics = metrics
        
        if seed:
            seed_everything(seed)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20,
                                      verbose=True, factor=0.2)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, masks = batch['images'], batch['masks']        
        logits = self(images)
        preds = torch.round(logits)
        loss = self.loss(logits, masks)
        
        logs = {}
        for k, metric in self.metrics.items():
            logs[k] = metric(preds, masks)

        out = {'loss': loss, 'log': logs}
        return out

    def training_epoch_end(self, outputs):
        dice_mean = 0
        loss_mean = 0
        n = len(outputs)
        for out in outputs:
            loss_mean += out['loss'].item()
            # TODO: generalize this instead of fixed `dice`
            dice_mean += out['log']['dice_coef']
        dice_mean /= n
        loss_mean /= n

        out_dicts = {'train_loss': loss_mean, 'dice': dice_mean} 
        results = {
            'log': out_dicts,
            'progress_bar': out_dicts
        }
        return results

    # VALIDATION LOOP
    def validation_step(self, batch, batch_idx):
        images, masks = batch['images'], batch['masks']
        logits = self(images)
        loss = self.loss(logits, masks)
        preds = torch.round(logits)
        
        logs = {}
        for k, metric in self.metrics.items():
            logs[f'val_{k}'] = metric(preds, masks)
        
        return {'val_loss': loss, 'log': logs}
    
    def validation_epoch_end(self, outputs):
        dice_mean = 0
        loss_mean = 0
        n = len(outputs)
        for out in outputs:
            loss_mean += out['val_loss']
            # TODO: generalize this instead of fixed `dice`
            dice_mean += out['log']['val_dice_coef']
        dice_mean /= n
        loss_mean /= n

        out_dicts = {'val_loss': loss_mean, 'val_dice': dice_mean}
        results = {
            'log': out_dicts,
            'progress_bar': out_dicts
        }
        return results

    def test_step(self, batch, batch_idx):
        images, masks = batch['images'], batch['masks']
        logits = self(images)
        preds = torch.round(logits)
        loss = self.loss(logits, masks).item()
        logs = {}
        for k, metric in self.metrics.items():
            logs[f'test_{k}'] = metric(preds, masks)

        return {'test_loss': loss, 'log': logs}

    def test_epoch_end(self, outputs):
        dice_mean = 0
        loss_mean = 0
        n = len(outputs)
        for out in outputs:
            loss_mean += out['test_loss']
            # TODO: generalize this instead of fixed `dice`
            dice_mean += out['log']['test_dice_coef']
        dice_mean /= n
        loss_mean /= n

        out_dicts = {'test_loss': loss_mean, 'test_dice': dice_mean}
        results = {
            'log': out_dicts,
            'progress_bar': out_dicts
        }
        return results
