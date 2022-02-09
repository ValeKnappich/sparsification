import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class SSTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, model_id):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.dataset_dict = datasets.load_dataset("sst").map(self.preprocess)
        self.dataset_dict.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        self.batch_size = batch_size

    def dataloader(self, split):
        return DataLoader(
            self.dataset_dict[split],
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=bool(split == "train"),
        )

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("validation")

    def test_dataloader(self):
        return self.dataloader("test")

    def preprocess(self, example):
        return self.tokenizer(example["sentence"], padding="max_length", max_length=65)
