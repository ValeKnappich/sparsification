import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError

from src.utils import utils

logging.basicConfig()
log = utils.get_logger(__name__)
log.setLevel(logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "true" if torch.cuda.is_available() else "false"


def benchmark(config: DictConfig):
    """
    Benchmark a trained model to measure runtime and accuracy metrics
    """
    # Init wandb
    log.info("Initializing wandb logger")
    wandb.init(**config.wandb, dir=(Path(config.work_dir) / "logs").resolve())

    # Prepare backend
    backend_cls = hydra.utils._locate(config.backend._target_)
    backend = backend_cls(config)
    backend.prepare()

    # Run prediction
    for dataset_split, dataloader_fn in (
        ("val", backend.datamodule.val_dataloader),
        ("test", backend.datamodule.test_dataloader),
    ):
        log.info(f"Starting Prediction of {dataset_split} data")

        # Create Metrics
        accuracy = Accuracy().to(backend.metric_device)
        mae = MeanAbsoluteError().to(backend.metric_device)
        mse = MeanSquaredError().to(backend.metric_device)
        n_batches = 0
        start_time = time.perf_counter()

        for batch in tqdm.tqdm(dataloader_fn()):
            # Separate input data and labels
            labels = batch["label"]
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            labels = labels.to(backend.metric_device, non_blocking=True)
            batch = {k: v for k, v in batch.items() if k != "label"}

            # Run prediction
            logits = backend.predict(batch)

            # Compute metrics
            accuracy(logits, torch.where(labels > 0.5, 1, 0))
            mae(logits, labels)
            mse(logits, labels)
            n_batches += 1

        end_time = time.perf_counter()

        wandb.log(
            {
                f"{dataset_split}_acc": accuracy.compute(),
                f"{dataset_split}_mse": mse.compute(),
                f"{dataset_split}_mae": mae.compute(),
                f"{dataset_split}_total_time (s)": end_time - start_time,
                f"{dataset_split}_n_batches": n_batches,
                f"{dataset_split}_time_per_batch (s)": (end_time - start_time) / n_batches,
                **OmegaConf.to_container(backend.config, resolve=True),
            }
        )
