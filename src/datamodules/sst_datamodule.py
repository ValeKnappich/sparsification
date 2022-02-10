import datasets
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class SSTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, model_id: str, tensor_format: str):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.dataset_dict = datasets.load_dataset("sst").map(self.preprocess)
        self.dataset_dict.set_format(
            tensor_format, columns=["input_ids", "attention_mask", "label"]
        )
        self.batch_size = batch_size
        self.tensor_format = tensor_format

    def dataloader(self, split: str):
        return DataLoader(
            self.dataset_dict[split],
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=bool(split == "train"),
            collate_fn=self.numpy_collate if self.tensor_format == "numpy" else None,
        )

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("validation")

    def test_dataloader(self):
        return self.dataloader("test")

    def preprocess(self, example):
        return self.tokenizer(example["sentence"], padding="max_length", max_length=65)

    def numpy_collate(self, examples):
        fields = list(examples[0].keys())
        return {field: np.stack([example[field] for example in examples]) for field in fields}
