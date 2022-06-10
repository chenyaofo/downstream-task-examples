
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule

from conf import config
from .model.resnet import models

def get_base_model():
    import dill
    model = torch.load(config["base_model"], pickle_module=dill)
    return model

def get_student_model(name, **kwargs):
    return models[name](**kwargs)

def dist_loss(teacher_out, student_out, T=1):
    prob_t = F.softmax(teacher_out/T, dim=1)
    log_prob_s = F.log_softmax(student_out/T, dim=1)
    dist_loss = -(prob_t*log_prob_s).sum(dim=1).mean()
    return dist_loss

class KnowledgeDistillation(LightningModule):
    def __init__(self, lr=0.05, student="cifar10_resnet20"):
        super().__init__()

        self.save_hyperparameters()
        self.base_model = get_base_model()
        self.student_model = get_student_model(student)

    def forward(self, x):
        out = self.student_model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        student_logits = self(x)
        with torch.no_grad():
            teacher_logits = self.base_model(x)
        loss = F.nll_loss(student_logits, y) + dist_loss(teacher_logits, student_logits)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.student_model.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // config["batch_size"]
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}