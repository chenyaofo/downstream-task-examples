import imp
import torch

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from module import LitResnet
from data import data_module

seed_everything(7)

AVAIL_GPUS = min(1, torch.cuda.device_count())


lighting_module = LitResnet(lr=0.05)
setattr(lighting_module, "datamodule", data_module)
# lighting_module.datamodule = data_module

trainer = Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=30,
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger("lightning_logs/", name="resnet"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
)

trainer.fit(lighting_module, data_module)
trainer.test(lighting_module, datamodule=data_module)