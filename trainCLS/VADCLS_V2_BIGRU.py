from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_s.VADdataset import VAD_DataSet_norm
from models.CVNT import CVNT_BiGRU
from trainCLS.baseCLS import BasicCLS
from lib.plot import spec_probs_to_figure

class VADCLS(BasicCLS):
    def __init__(self, config):
        super().__init__(config=config)
        self.model = CVNT_BiGRU(config,output_size=1)

        self.train_dataset = VAD_DataSet_norm(config=config, infer=False)
        self.val_dataset = VAD_DataSet_norm(config=config, infer=True)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, mask=None):
        x = self.model(x, mask)

        return x

    # def before_opt(self):
    #     sc=[]
    #     for p in self.model.parameters():
    #         if p.grad is not None:
    #             sc.append(torch.norm(p.grad) ** 2)
    #     if sc!=[]:
    #         self.grad_norm = torch.norm(torch.stack(sc) ** 0.5)

    def training_step(self, batch, batch_idx: int, ):
        if batch['mask'] is not None:
            batch['mask']=(batch['mask']==0)
        P = self.forward(batch['mel'], batch['mask'])
        losses = self.loss(P, batch['target'].float()[:,None,:])
        lr = self.lrs.get_last_lr()[0]
        if self.opt_step % 50 == 0:
            tb_log = {}
            tb_log['training/loss'] = losses
            # tb_log['training/accuracy_train'] = accuracy_train
            tb_log['training/lr'] = lr
            tb_log['training/grad_norm'] = self.grad_norm
            self.logger.log_metrics(tb_log, step=self.opt_step)
        return {'loss': losses, 'logges': {'l2loss': losses, 'lr': lr,'gn':self.grad_norm}}

    def on_sum_validation_logs(self, logs):
        # self.lrs.set_step(self.opt_step)
        tb_log = {'val/loss': logs['val_loss']}
        self.logger.log_metrics(tb_log, step=self.opt_step)

    def validation_step(self, batch, batch_idx: int):
        P = self.forward(batch['mel'], batch['mask'])
        losses = self.loss(P, batch['target'].float()[:,None,:])


        name_prefix = 'mel'
        P=torch.sigmoid(P)
        P=(P>self.config['prob_voice']).long()[0].cpu()[0]
        fig = spec_probs_to_figure(spec=batch['mel'][0].cpu(), prob_gt=batch['target'].long()[0].cpu(),
                                   prob_pred=P)
        self.logger.experiment.add_figure(f'{name_prefix}_{batch_idx}', fig, global_step=self.opt_step)

        return {'val_loss': losses}
