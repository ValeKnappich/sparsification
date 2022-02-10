from abc import ABC, abstractmethod

import torch


class BackendBase(ABC):
    @abstractmethod
    def prepare(self) -> None:
        """
        Prepare model, datamodule and runtime
        """

    @abstractmethod
    def predict(self, batch: dict) -> torch.Tensor:
        """
        Run prediction on the backend for a given batch
        """
