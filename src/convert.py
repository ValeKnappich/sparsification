import logging
import time
from pathlib import Path

import hydra
import onnxruntime.quantization as ort_quant
import torch
from omegaconf import DictConfig

from src.models import SSTModel
from src.utils import utils

logging.basicConfig()
log = utils.get_logger(__name__)
log.setLevel(logging.INFO)


def convert_to_onnx(config: DictConfig):
    # Load model
    checkpoint_path = (Path(config.work_dir) / config.checkpoint_path).resolve()
    model = load_lightning(checkpoint_path)

    out_path = (Path(config.work_dir) / config.checkpoint_path).with_suffix(".onnx")
    if out_path.exists():
        # Add timestamp to filename
        out_path = Path(f"{out_path.with_suffix('')}-{int(time.time())}{out_path.suffix}")
    log.info(f"Saving ONNX model to path {out_path}")

    input_sample = {
        "input_ids": torch.randint(0, 100, (1, config.sequence_length), dtype=torch.int32),
        "attention_mask": torch.ones(1, config.sequence_length, dtype=torch.int8),
    }
    model.to_onnx(out_path, input_sample, export_params=True, opset_version=14)


def quantize_onnx(config: DictConfig):
    checkpoint_path = (Path(config.work_dir) / config.checkpoint_path).resolve()
    out_path = Path(f"{checkpoint_path.with_suffix('')}-quant.onnx")
    if out_path.exists():
        # Add timestamp to filename
        out_path = Path(f"{out_path.with_suffix('')}-{int(time.time())}{out_path.suffix}")
    log.info(f"Saving ONNX model to path {out_path}")
    ort_quant.quantize_dynamic(checkpoint_path, out_path)


def load_lightning(checkpoint_path: Path):
    log.info("Loading model")
    return SSTModel.load_from_checkpoint(checkpoint_path)


def convert(config: DictConfig):
    """
    Benchmark a trained model to measure runtime and accuracy metrics
    """
    convert_fn = hydra.utils._locate(config.conversion._target_)
    return convert_fn(config)
