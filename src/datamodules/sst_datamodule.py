import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class SSTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, model_id):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.dataset_dict = (
            datasets.load_dataset("sst").map(
                lambda x: self.tokenizer(x["sentence"], padding="max_length", max_length=65)
            )
            # .map(
            #     # Add or remove epsilone if labels are 1 or 0 to avoid error in BCELoss
            #     lambda x: dict(
            #         label=x["label"] - 1e-6
            #         if x["label"] == 1
            #         else x["label"] + 1e-6
            #         if x["label"] == 0
            #         else x["label"],
            #         **{k: v for k, v in x.items() if k != "label"}
            #     )
            # )
        )

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
