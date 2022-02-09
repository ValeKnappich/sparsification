import logging
import os
import time
from pathlib import Path
from typing import Callable

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.models import SSTModel
from src.utils import utils

logging.basicConfig()
log = utils.get_logger(__name__)
log.setLevel(logging.INFO)
project_root = Path(__file__).parent.parent
os.environ["TOKENIZERS_PARALLELISM"] = "true" if torch.cuda.is_available() else "false"

conversion_registry = {}


def register_conversion(key: str):
    def register_conversion_inner(f: Callable):
        if key in conversion_registry:
            log.warning(f"Multiple functions registered for key '{key}'")
        conversion_registry[key] = f
        return f

    return register_conversion_inner


@register_conversion("onnx")
def convert_to_onnx(model: pl.LightningModule, config: DictConfig):
    out_path = (project_root / config.conversion.checkpoint_path).with_suffix(".onnx")
    if out_path.exists():
        # Add timestamp to filename
        out_path = Path(f"{out_path.with_suffix('')}-{int(time.time())}{out_path.suffix}")
    log.info(f"Saving ONNX model to path {out_path}")

    input_sample = {
        field: torch.randn((1, config.conversion.sequence_length))
        for field in ("input_ids", "attention_mask")
    }
    input_sample = {
        "input_ids": torch.randint(0, 100, (1, config.conversion.sequence_length)),
        "attention_mask": torch.zeros(1, config.conversion.sequence_length),
    }
    model.to_onnx(out_path, input_sample, export_params=True, opset_version=11)


def convert(config: DictConfig):
    """
    Benchmark a trained model to measure runtime and accuracy metrics
    """
    # Init lightning model
    log.info("Loading model")
    checkpoint_path = (project_root / config.conversion.checkpoint_path).resolve()
    model: pl.LightningModule = SSTModel.load_from_checkpoint(checkpoint_path)

    # Run conversion
    convert_fn = conversion_registry[config.conversion.key]
    return convert_fn(model, config)
