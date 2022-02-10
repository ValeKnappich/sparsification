import logging
from pathlib import Path

import hydra
import onnxruntime as ort
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.backend.base import BackendBase
from src.utils import utils

logging.basicConfig()
log = utils.get_logger(__name__)
log.setLevel(logging.INFO)


class ONNXBackend(BackendBase):
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def prepare(self) -> None:
        if not self.config.datamodule.model_id:
            raise Exception(
                "model_id could not be infered to get correct tokenizer."
                "Please set datamodule.model_id"
            )

        # Init lightning datamodule
        if self.config.datamodule.tensor_format != "numpy":
            log.warning(
                "Invalid tensor format for ONNX backend. Setting 'datamodule.tensor_format' "
                f"from '{self.config.datamodule.tensor_format}' to 'numpy'"
            )
            self.config.datamodule.tensor_format = "numpy"
        log.info(f"Instantiating datamodule <{self.config.datamodule._target_}>")
        self.datamodule: pl.LightningDataModule = hydra.utils.instantiate(self.config.datamodule)

        # Prepare device
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            log.warning("Cuda is not available. Using CPU")
            self.device = torch.device("cpu")
            self.config.device = self.device.type
        # logits will be on cpu anyway, so always compute metrics on cpu to avoid memory movement
        self.metric_device = torch.device("cpu")
        log.info(f"Running benchmark on {self.device.type}")

        # Create session
        sess_options = ort.SessionOptions()
        model_path = (Path(self.config.work_dir) / self.config.checkpoint_path).resolve()
        if self.device.type == "cpu":
            self.session = ort.InferenceSession(
                str(model_path), sess_options, providers=["CPUExecutionProvider"]
            )
        else:
            self.session = ort.InferenceSession(
                str(model_path), sess_options, providers=["CUDAExecutionProvider"]
            )

    def predict(self, batch: dict) -> torch.Tensor:
        logits = self.session.run(None, batch)
        return torch.from_numpy(logits[0])
