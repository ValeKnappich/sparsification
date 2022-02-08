import os
import time
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError

from src.utils import utils

log = utils.get_logger(__name__)
project_root = Path(__file__).parent.parent
os.environ["TOKENIZERS_PARALLELISM"] = "true" if torch.cuda.is_available() else "false"


def move_tensors_to_device(data: dict, device: torch.device):
    """
    Moves all tensors in a dict to a device
    """
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = data[key].to(device)
        elif isinstance(value, dict):
            move_tensors_to_device(value, device)


def benchmark(config: DictConfig):
    """
    Benchmark a trained model to measure runtime and accuracy metrics
    """

    # Pass model_id from model config to datamodule config to get correct tokenizer
    config.datamodule["model_id"] = config.model.model_id

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model_cls: type = hydra.utils._locate(config.model._target_)
    checkpoint_path: Path = (project_root / config.benchmark.checkpoint_path).resolve()
    model: pl.LightningModule = model_cls.load_from_checkpoint(checkpoint_path)

    # Init wandb
    log.info("Initializing wandb logger")
    wandb.init(**config.wandb, dir=(project_root / "logs").resolve())

    # Prepare device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        log.warning("Cuda is not available. Using CPU")
    wandb.log(dict(device=device.type))

    # Prepare model
    log.info("Preparing model for prediction")
    torch.set_grad_enabled(False)
    model.eval()
    model.to(device)

    # Run prediction
    for dataset_split, dataloader_fn in (
        ("val", datamodule.val_dataloader),
        ("test", datamodule.test_dataloader),
    ):

        log.info(f"Starting Prediction of {dataset_split} data")

        # Create Metrics
        accuracy = Accuracy().to(device)
        mae = MeanAbsoluteError().to(device)
        mse = MeanSquaredError().to(device)

        start_time = time.time()

        for batch in tqdm.tqdm(dataloader_fn()):
            move_tensors_to_device(batch, device)

            # Run prediction
            logits = model(batch)

            # Compute metrics
            accuracy(logits, torch.where(batch["label"] > 0.5, 1, 0))
            mae(logits, batch["label"])
            mse(logits, batch["label"])

        end_time = time.time()

        wandb.log(
            {
                f"{dataset_split}_acc": accuracy.compute(),
                f"{dataset_split}_mse": mse.compute(),
                f"{dataset_split}_mae": mae.compute(),
                f"{dataset_split}_total_time": end_time - start_time,
                **OmegaConf.to_container(config, resolve=True),
            }
        )
