import pytorch_lightning as pl
import streamlit as st
import transformers


def build_app(model: pl.LightningModule):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model.model_id)

    st.write("# Sentiment Analysis")
    st.info("The model used here is `distilroberta-base` trained on the `SST` dataset.")

    text_input = st.text_input("Text Input", "This is great")

    with st.spinner(text="Processing text ..."):
        tokens = tokenizer(text_input, return_tensors="pt")
        out = model(tokens)

    st.write(out.item())

    import subprocess

    st.write(subprocess.check_output(["cat", "/proc/cpuinfo"]))


if __name__ == "__main__":
    from pathlib import Path

    from src.models import SSTModel

    model = SSTModel.load_from_checkpoint(
        Path(__file__).parent.parent.parent / "checkpoints/distilroberta-base.ckpt"
    )
    build_app(model)
