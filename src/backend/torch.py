import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.backend.base import BackendBase
from src.models.sst_model import SSTModel
from src.utils import utils

logging.basicConfig()
log = utils.get_logger(__name__)
log.setLevel(logging.INFO)


class TorchBackend(BackendBase):
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def prepare(self) -> None:
        # Init lightning model
        log.info("Loading model")
        checkpoint_path: Path = (
            Path(self.config.work_dir) / self.config.checkpoint_path
        ).resolve()
        self.model: pl.LightningModule = SSTModel.load_from_checkpoint(checkpoint_path)

        # Pass model_id from model to datamodule config to get correct tokenizer
        self.config.datamodule["model_id"] = self.model.model_id

        # Init lightning datamodule
        log.info(f"Instantiating datamodule <{self.config.datamodule._target_}>")
        self.datamodule: pl.LightningDataModule = hydra.utils.instantiate(self.config.datamodule)

        # Prepare device
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            log.warning("Cuda is not available. Using CPU")
            self.device = torch.device("cpu")
            self.config.device = (
                self.device.type
            )  # make sure logger receives correct device type in case of change
        self.metric_device = self.device
        log.info(f"Running benchmark on {self.device.type}")

        # Prepare model
        log.info("Preparing model for prediction")
        torch.set_grad_enabled(False)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, batch: dict) -> torch.Tensor:
        self.move_tensors_to_device(batch)
        return self.model(batch)

    def move_tensors_to_device(self, data: dict) -> None:
        """
        Moves all tensors in a dict to a device
        """
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = data[key].to(self.device)
            elif isinstance(value, dict):
                self.move_tensors_to_device(value, self.device)
