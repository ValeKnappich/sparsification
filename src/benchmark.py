from pathlib import Path
import time

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import wandb

from src.utils import utils

log = utils.get_logger(__name__)
project_root = Path(__file__).parent.parent


def benchmark(config: DictConfig):
    """
    Benchmark a trained model to measure runtime and accuracy metrics
    """
    
    # Pass model_id from model config to datamodule config to get correct tokenizer
    config.datamodule["model_id"] = config.model.model_id
    
    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")  
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    
    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model_cls: type = hydra.utils._locate(config.model._target_)
    checkpoint_path: Path = (project_root / config.benchmark.checkpoint_path).resolve()
    model: LightningModule = model_cls.load_from_checkpoint(checkpoint_path)
    
    # Init wandb
    wandb.init(
        **config.wandb, config=config, 
        dir=(project_root / "logs").resolve()
    )
    
    # Prepare device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        log.warning("cuda is not available. Using CPU")
    wandb.log(dict(device=device.type))
    
    # Run prediction
    for dataset_split, dataloader_fn in (("val", datamodule.val_dataloader), 
                                      ("test", datamodule.test_dataloader)):
        
        model.to(device)
        start_time: float = time.time()
        
        for batch in dataloader_fn():
            print(batch)
            break
            
        
        