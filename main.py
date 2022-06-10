import torch

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from custom.module import KnowledgeDistillation
from custom.data import data_module

from conf import config

seed_everything(config["seed"])

lighting_module = KnowledgeDistillation()
setattr(lighting_module, "datamodule", data_module)

trainer = Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=config["max_epochs"],
    gpus=config["num_gpus_per_nodes"],
    logger=TensorBoardLogger("lightning_logs/", name="resnet"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
)

trainer.fit(lighting_module, data_module)
trainer.test(lighting_module, datamodule=data_module)