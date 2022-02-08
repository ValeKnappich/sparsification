from abc import ABC, abstractmethod

import torch


class ModelAccelerator(ABC):
    @abstractmethod
    def prepare_model() -> torch.Module:
        """Prepare model for efficient inference, e.g. by quantizing the weights"""
