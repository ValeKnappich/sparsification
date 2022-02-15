import pytorch_lightning as pl
import streamlit as st


def build_app(model: pl.LightningModel):
    st.write("# Hello World")


if __name__ == "__main__":
    from pathlib import Path

    from src.models import SSTModel

    model = SSTModel.load_from_checkpoint(
        Path(__file__).parent.parent.parent / "checkpoints/distilroberta-base.ckpt"
    )
    build_app(model)
